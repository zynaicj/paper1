# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils.freq_fusion import FreqFusion
from mmseg.registry import MODELS


@MODELS.register_module()
class FreqFusionDecoder(BaseDecodeHead):
    """SegFormer decoder with cascade FreqFusion multi-scale fusion."""

    def __init__(self, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(self.in_channels) == 4

        self.input_convs = nn.ModuleList([
            ConvModule(
                in_channels=in_channel,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg) for in_channel in self.in_channels
        ])

        self.ff_c4_c3 = self._build_freq_fusion()
        self.ff_f3_c2 = self._build_freq_fusion()
        self.ff_f2_c1 = self._build_freq_fusion()

        self.fusion_conv = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _build_freq_fusion(self):
        return FreqFusion(
            hr_channels=self.channels,
            lr_channels=self.channels,
            scale_factor=1,
            lowpass_kernel=5,
            highpass_kernel=3,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64,
            align_corners=self.align_corners,
            upsample_mode='nearest',
            feature_resample=False,
            comp_feat_upsample=True,
            use_high_pass=True,
            use_low_pass=True,
            hr_residual=True,
            semi_conv=True,
            hamming_window=True)

    @staticmethod
    def _resize_like(feat, ref):
        if feat.shape[2:] != ref.shape[2:]:
            feat = F.interpolate(feat, size=ref.shape[2:], mode='nearest')
        return feat

    def _fuse(self, freq_fusion, hr_feat, lr_feat):
        _, hr_feat, lr_feat = freq_fusion(hr_feat=hr_feat, lr_feat=lr_feat)
        lr_feat = self._resize_like(lr_feat, hr_feat)
        return hr_feat + lr_feat

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        c1, c2, c3, c4 = inputs

        c1_proj = self.input_convs[0](c1)
        c2_proj = self.input_convs[1](c2)
        c3_proj = self.input_convs[2](c3)
        c4_proj = self.input_convs[3](c4)

        f3 = self._fuse(self.ff_c4_c3, c3_proj, c4_proj)
        f2 = self._fuse(self.ff_f3_c2, c2_proj, f3)
        f1 = self._fuse(self.ff_f2_c1, c1_proj, f2)

        out = self.fusion_conv(f1)
        out = self.cls_seg(out)
        return out
