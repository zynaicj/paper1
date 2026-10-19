# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils.freq_fusion import FreqFusion
from mmseg.registry import MODELS
from ..utils import resize


class CBAMGate(nn.Module):
    """CBAM refinement for the frequency residual gate feature.

    The module applies channel and spatial attention to the gate feature that
    is produced from ``[base_hr, freq_feat, freq_delta]``. It only improves the
    residual selection branch and does not replace the decoder feature itself.
    """

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        assert spatial_kernel % 2 == 1
        hidden_channels = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True))
        self.spatial_conv = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=True)

    def forward(self, x):
        avg_attn = self.channel_mlp(F.adaptive_avg_pool2d(x, 1))
        max_attn = self.channel_mlp(F.adaptive_max_pool2d(x, 1))
        channel_attn = torch.sigmoid(avg_attn + max_attn)
        x = x * channel_attn

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        spatial_attn = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        return x * spatial_attn


@MODELS.register_module()
class SegformerGatedFreqFusionHead(BaseDecodeHead):
    """SegFormer head with gated residual FreqFusion injection.

    This head keeps the original SegFormerHead concat-fusion path as the main
    branch. FreqFusion is used as an auxiliary frequency branch and its output
    is injected back by a learnable gated residual. The residual projection is
    initialized to zero, so the head starts close to the baseline SegFormerHead
    and learns to use frequency information only when it helps.
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
                 gate_channels=64,
                 residual_scale=1.0,
                 use_high_pass=True,
                 use_low_pass=True,
                 gate_type='spatial',
                 gate_bias_init=0.0,
                 cbam_reduction=16,
                 cbam_spatial_kernel=7,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        assert len(self.in_channels) == len(self.in_index)
        assert freqfusion_pair in self._FUSION_PAIRS
        assert gate_type in ('spatial', 'channel', 'cbam')
        self.interpolate_mode = interpolate_mode
        self.freqfusion_pair = freqfusion_pair
        self.compressed_channels = compressed_channels
        self.residual_scale = residual_scale
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.gate_type = gate_type
        self.gate_bias_init = gate_bias_init
        self.cbam_reduction = cbam_reduction
        self.cbam_spatial_kernel = cbam_spatial_kernel

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
            use_high_pass=use_high_pass,
            use_low_pass=use_low_pass,
            hr_residual=True,
            semi_conv=True,
            hamming_window=True)

        fusion_channels = self.channels * 3
        self.residual_refine = ConvModule(
            in_channels=fusion_channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.residual_proj = nn.Conv2d(self.channels, self.channels, 1)

        if gate_type == 'cbam':
            # CBAM refines the gate feature for channel-spatial selection of
            # FreqFusion residuals; the main SegFormer fusion path is unchanged.
            self.gate_embed = ConvModule(
                in_channels=fusion_channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.cbam_gate = CBAMGate(
                self.channels,
                reduction=cbam_reduction,
                spatial_kernel=cbam_spatial_kernel)
            self.gate_pred = nn.Conv2d(
                self.channels, self.channels, kernel_size=3, padding=1)
        else:
            self.gate_embed = ConvModule(
                in_channels=fusion_channels,
                out_channels=gate_channels,
                kernel_size=1,
                norm_cfg=None,
                act_cfg=self.act_cfg)
            self.cbam_gate = None
            gate_out_channels = 1 if gate_type == 'spatial' else self.channels
            self.gate_pred = nn.Conv2d(
                gate_channels, gate_out_channels, kernel_size=3, padding=1)

        self.fusion_conv = ConvModule(
            in_channels=self.channels * len(self.in_channels),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def init_weights(self):
        super().init_weights()
        nn.init.zeros_(self.residual_proj.weight)
        if self.residual_proj.bias is not None:
            nn.init.zeros_(self.residual_proj.bias)
        nn.init.zeros_(self.gate_pred.weight)
        if self.gate_pred.bias is not None:
            nn.init.constant_(self.gate_pred.bias, self.gate_bias_init)

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
        base_hr = feats[hr_idx]
        _, hr_feat, lr_feat = self.freq_fusion(
            hr_feat=base_hr, lr_feat=feats[lr_idx])
        lr_feat = self._resize_like(lr_feat, hr_feat)

        freq_feat = hr_feat + lr_feat
        freq_delta = freq_feat - base_hr
        fusion_input = torch.cat([base_hr, freq_feat, freq_delta], dim=1)

        gate_feat = self.gate_embed(fusion_input)
        if self.gate_type == 'cbam':
            gate_feat = self.cbam_gate(gate_feat)
        gate = torch.sigmoid(self.gate_pred(gate_feat))
        residual = self.residual_proj(self.residual_refine(fusion_input))
        feats[hr_idx] = base_hr + self.residual_scale * gate * residual

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


@MODELS.register_module()
class SegformerSemanticGuidedFreqFusionHead(BaseDecodeHead):
    """SegFormer head with semantic-guided gated FreqFusion residual.

    The original SegFormer multi-level concat-fusion branch is kept as the main
    decoder path. FreqFusion is only used to produce a residual at one adjacent
    feature pair, while the gate additionally sees high-level semantic context
    from a deeper stage. This lets semantic cues decide where and which channels
    should accept frequency residuals, without replacing the decoder features.
    """

    _FUSION_PAIRS = {
        'c4_c3': (2, 3),
        'c3_c2': (1, 2),
        'c2_c1': (0, 1),
    }

    def __init__(self,
                 interpolate_mode='bilinear',
                 freqfusion_pair='c2_c1',
                 semantic_index=3,
                 compressed_channels=64,
                 residual_scale=1.0,
                 use_high_pass=True,
                 use_low_pass=True,
                 gate_bias_init=0.0,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        assert len(self.in_channels) == len(self.in_index)
        assert freqfusion_pair in self._FUSION_PAIRS
        assert 0 <= semantic_index < len(self.in_channels)
        self.interpolate_mode = interpolate_mode
        self.freqfusion_pair = freqfusion_pair
        self.semantic_index = semantic_index
        self.compressed_channels = compressed_channels
        self.residual_scale = residual_scale
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.gate_bias_init = gate_bias_init

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
            use_high_pass=use_high_pass,
            use_low_pass=use_low_pass,
            hr_residual=True,
            semi_conv=True,
            hamming_window=True)

        fusion_channels = self.channels * 3
        gate_channels = self.channels * 4

        self.semantic_refine = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.residual_refine = ConvModule(
            in_channels=fusion_channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.residual_proj = nn.Conv2d(self.channels, self.channels, 1)

        # The semantic context is used only to generate the channel-wise gate
        # for selecting FreqFusion residuals, not to overwrite decoder features.
        self.gate_embed = ConvModule(
            in_channels=gate_channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.gate_pred = nn.Conv2d(
            self.channels, self.channels, kernel_size=3, padding=1)

        self.fusion_conv = ConvModule(
            in_channels=self.channels * len(self.in_channels),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def init_weights(self):
        super().init_weights()
        nn.init.zeros_(self.residual_proj.weight)
        if self.residual_proj.bias is not None:
            nn.init.zeros_(self.residual_proj.bias)
        nn.init.zeros_(self.gate_pred.weight)
        if self.gate_pred.bias is not None:
            nn.init.constant_(self.gate_pred.bias, self.gate_bias_init)

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
        base_hr = feats[hr_idx]
        _, hr_feat, lr_feat = self.freq_fusion(
            hr_feat=base_hr, lr_feat=feats[lr_idx])
        lr_feat = self._resize_like(lr_feat, hr_feat)

        freq_feat = hr_feat + lr_feat
        freq_delta = freq_feat - base_hr
        fusion_input = torch.cat([base_hr, freq_feat, freq_delta], dim=1)

        semantic_context = self.semantic_refine(feats[self.semantic_index])
        semantic_context = resize(
            input=semantic_context,
            size=base_hr.shape[2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners)

        gate_input = torch.cat([fusion_input, semantic_context], dim=1)
        gate = torch.sigmoid(self.gate_pred(self.gate_embed(gate_input)))
        residual = self.residual_proj(self.residual_refine(fusion_input))
        feats[hr_idx] = base_hr + self.residual_scale * gate * residual

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
