# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .segformer_residual_aspp_head import SegformerResidualASPPHead


class FrequencyDecoupledFuse(nn.Module):
    """Frequency-decoupled cross-scale feature fusion.

    Motivated by underwater light propagation: scattering and absorption act as
    a low-pass filter on the scene radiance, suppressing high-frequency detail
    (boundaries, texture) while preserving low-frequency structure (semantic
    regions). This module decomposes the fusion of an adjacent decoder feature
    pair into two independent paths:

    - **Low-frequency path**: injects semantic context from the higher-level
      guidance feature into the low-frequency component of the target feature.
    - **High-frequency path**: enhances boundary and texture detail by fusing
      high-frequency components from both the target and guidance features.

    Both paths are zero-initialized so the module starts as an identity mapping,
    compatible with loading a pre-trained A+B checkpoint.

    Args:
        channels (int): Input/output channels after SegFormer 1x1 projection.
        mid_channels (int): Hidden channels for relation encoding.
        init_gamma_low (float): Initial value for the low-frequency residual
            scale. Zero means the path starts as identity.
        init_gamma_high (float): Initial value for the high-frequency residual
            scale. Zero means the path starts as identity.
        sim_type (str): ``'spatial'`` produces [B,1,H,W] gates and
            ``'channel'`` produces [B,C,H,W] gates.
        align_corners (bool): Passed to ``resize``.
        conv_cfg (dict | None): Config for ConvModule.
        norm_cfg (dict): Config for ConvModule normalization.
        act_cfg (dict): Config for ConvModule activation.
    """

    def __init__(self,
                 channels=256,
                 mid_channels=128,
                 init_gamma_low=0.0,
                 init_gamma_high=0.0,
                 sim_type='spatial',
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        if sim_type not in ('spatial', 'channel'):
            raise ValueError(
                f"sim_type must be 'spatial' or 'channel', got {sim_type}.")

        self.channels = channels
        self.mid_channels = mid_channels
        self.sim_type = sim_type
        self.align_corners = align_corners

        # -- Frequency decomposition --
        # 3x3 average pooling serves as a low-pass filter. The high-frequency
        # residual is obtained by subtracting the low-pass output.
        self.low_pass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # -- Relation encoding --
        # Projects both features into a shared space, then encodes their
        # agreement (product) and disagreement (absolute difference).
        self.f_x = ConvModule(
            channels,
            mid_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.f_y = ConvModule(
            channels,
            mid_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.relation_encoder = ConvModule(
            mid_channels * 2,
            mid_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # -- Gate predictors --
        gate_channels = 1 if sim_type == 'spatial' else channels
        self.low_gate = nn.Conv2d(mid_channels, gate_channels, kernel_size=1)
        self.high_gate = nn.Conv2d(mid_channels, gate_channels, kernel_size=1)

        # -- Detail enhancement for the high-frequency path --
        # Depthwise 3x3 conv refines the target's high-frequency detail;
        # a 1x1 conv brings guidance high-frequency into the same space.
        self.detail_enhance = nn.Sequential(
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None))
        self.high_project = ConvModule(
            channels,
            channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # -- Zero-initialized residual scales --
        self.gamma_low = nn.Parameter(torch.tensor(float(init_gamma_low)))
        self.gamma_high = nn.Parameter(torch.tensor(float(init_gamma_high)))

    def forward(self, x, y):
        """Fuse guidance feature ``y`` into target feature ``x``.

        Args:
            x (Tensor): Target feature [B, C, H, W] (shallower decoder stage).
            y (Tensor): Guidance feature [B, C, H', W'] (deeper decoder stage).

        Returns:
            Tensor: Fused feature [B, C, H, W].
        """
        # 1. Align spatial resolution.
        y_up = resize(
            input=y,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # 2. Frequency decomposition.
        x_low = self.low_pass(x)
        x_high = x - x_low
        y_low = self.low_pass(y_up)
        y_high = y_up - y_low

        # 3. Relation encoding.
        x_proj = self.f_x(x)
        y_proj = self.f_y(y_up)
        relation = torch.cat([x_proj * y_proj, torch.abs(x_proj - y_proj)],
                             dim=1)
        relation = self.relation_encoder(relation)

        # 4. Low-frequency path — semantic injection.
        low_gate = torch.sigmoid(self.low_gate(relation))
        low_delta = low_gate * (y_low - x_low)

        # 5. High-frequency path — detail enhancement.
        high_gate = torch.sigmoid(self.high_gate(relation))
        detail_feat = self.detail_enhance(x_high) + self.high_project(y_high)
        high_delta = high_gate * detail_feat

        # 6. Zero-init residual update.
        return x + self.gamma_low * low_delta + self.gamma_high * high_delta


@MODELS.register_module()
class SegformerResidualASPPFDFHead(SegformerResidualASPPHead):
    """Residual-ASPP decoder with frequency-decoupled fusion (FDF).

    FDF is inserted after each stage's 1x1 projection and before the
    resize/concat multi-scale fusion. The default stage list fuses c4->c3 and
    c3->c2, leaving c1 and c4 unchanged::

        feats[2] = FDF(feats[2], feats[3])   # c3 enriched by c4
        feats[1] = FDF(feats[1], feats[2])   # c2 enriched by c3'

    All FDF paths are zero-initialized, so the module starts as an identity
    mapping from the pre-trained A+B checkpoint.

    This head keeps Module B (Residual ASPP) unchanged and only replaces
    Module C with frequency-decoupled progressive fusion.
    """

    _FUSE_STAGE_PAIRS = {
        'c4_c3': (2, 3),
        'c3_c2': (1, 2),
        'c2_c1': (0, 1),
    }

    def __init__(self,
                 fdf_channels=256,
                 fdf_mid_channels=128,
                 fdf_init_gamma_low=0.0,
                 fdf_init_gamma_high=0.0,
                 fdf_fuse_stages=('c4_c3', 'c3_c2'),
                 fdf_sim_type='spatial',
                 **kwargs):
        super().__init__(**kwargs)

        if fdf_channels != self.channels:
            raise ValueError(
                f'fdf_channels ({fdf_channels}) must match decode channels '
                f'({self.channels}) because FDF runs after projection.')
        if isinstance(fdf_fuse_stages, str):
            fdf_fuse_stages = (fdf_fuse_stages, )
        if fdf_fuse_stages is None:
            fdf_fuse_stages = ()

        invalid_stages = [
            stage for stage in fdf_fuse_stages
            if stage not in self._FUSE_STAGE_PAIRS
        ]
        if invalid_stages:
            valid = tuple(self._FUSE_STAGE_PAIRS)
            raise ValueError(
                f'Invalid fdf_fuse_stages {invalid_stages}; valid values '
                f'are {valid}.')

        self.fdf_fuse_stages = tuple(fdf_fuse_stages)
        self.fdf_fuses = nn.ModuleDict()
        for stage in self.fdf_fuse_stages:
            self.fdf_fuses[stage] = FrequencyDecoupledFuse(
                channels=fdf_channels,
                mid_channels=fdf_mid_channels,
                init_gamma_low=fdf_init_gamma_low,
                init_gamma_high=fdf_init_gamma_high,
                sim_type=fdf_sim_type,
                align_corners=self.align_corners,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _apply_fdf(self, feats):
        """Apply frequency-decoupled fusion in stage order."""
        for stage in self.fdf_fuse_stages:
            target_idx, guidance_idx = self._FUSE_STAGE_PAIRS[stage]
            feats[target_idx] = self.fdf_fuses[stage](
                feats[target_idx], feats[guidance_idx])
        return feats

    def forward(self, inputs):
        """Forward pass.

        Steps:
        1. 1x1 projection of each backbone stage feature.
        2. FDF progressive fusion (c4->c3, c3->c2).
        3. Resize all stages to c1 resolution and concat.
        4. Residual ASPP context refinement (Module B, unchanged).
        5. Optional post-processing modules (UPCM, RCM, LSK).
        6. Classification.
        """
        inputs = self._transform_inputs(inputs)
        feats = [conv(x) for conv, x in zip(self.convs, inputs)]
        feats = self._apply_fdf(feats)

        outs = [
            resize(
                input=feat,
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners) for feat in feats
        ]

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.residual_aspp(out, align_corners=self.align_corners)
        if self.upcm is not None:
            out = self.upcm(out)
        if self.rcm is not None:
            out = out + self.gamma_rcm * self.rcm(out)
        if self.lsk is not None:
            out = out + self.gamma_lsk * self.lsk(out)
        return self.cls_seg(out)
