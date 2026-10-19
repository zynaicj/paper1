# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .segformer_head import SegformerHead
from .segformer_residual_aspp_head import SegformerResidualASPPHead


class SemanticGuidedFuse(nn.Module):
    """Semantic-guided fusion for one adjacent decoder feature pair.

    The projected-product gate and convex feature blending are adapted from
    PIDNet's PagFM. This variant learns the similarity projection, adds a
    zero-initialized residual scale, and is applied progressively between
    adjacent SegFormer decoder stages.

    Args:
        channels (int): Input/output channels after SegFormer 1x1 projection.
        mid_channels (int): Hidden channels for similarity projection.
        init_gamma (float): Initial residual scale. A zero value makes the
            module start as an identity mapping.
        sim_type (str): ``spatial`` produces [B, 1, H, W] guidance and
            ``channel`` produces [B, C, H, W] guidance.
    """

    def __init__(self,
                 channels=256,
                 mid_channels=128,
                 init_gamma=0.0,
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

        sim_channels = 1 if sim_type == 'spatial' else channels
        self.sim_conv = nn.Conv2d(mid_channels, sim_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x, y):
        y_up = resize(
            input=y,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        x_proj = self.f_x(x)
        y_proj = self.f_y(y_up)
        sim = torch.sigmoid(self.sim_conv(x_proj * y_proj))
        fused = (1 - sim) * x + sim * y_up
        return x + self.gamma * (fused - x)


class BoundaryPreservingSemanticFuse(nn.Module):
    """Boundary-preserving semantic fusion for adjacent decoder features.

    This unit keeps the stable semantic injection path of SG-PCF, but predicts
    gates from both feature agreement and feature difference. A second branch
    writes back target-layer high-frequency details, which is useful for
    underwater scenes where high-level semantic injection can wash out object
    boundaries and seafloor texture.
    """

    def __init__(self,
                 channels=256,
                 mid_channels=128,
                 init_semantic_gamma=0.0,
                 init_detail_gamma=0.0,
                 sim_type='spatial',
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        if sim_type not in ('spatial', 'channel'):
            raise ValueError(
                f"sim_type must be 'spatial' or 'channel', got {sim_type}.")

        self.align_corners = align_corners
        self.sim_type = sim_type

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

        gate_channels = 1 if sim_type == 'spatial' else channels
        self.semantic_gate = nn.Conv2d(
            mid_channels, gate_channels, kernel_size=1)
        self.detail_gate = nn.Conv2d(
            mid_channels, gate_channels, kernel_size=1)
        self.detail_transform = nn.Sequential(
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

        self.semantic_gamma = nn.Parameter(
            torch.tensor(float(init_semantic_gamma)))
        self.detail_gamma = nn.Parameter(
            torch.tensor(float(init_detail_gamma)))

    def forward(self, x, y):
        y_up = resize(
            input=y,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        x_proj = self.f_x(x)
        y_proj = self.f_y(y_up)
        relation = torch.cat(
            [x_proj * y_proj, torch.abs(x_proj - y_proj)], dim=1)
        relation = self.relation_encoder(relation)

        semantic_gate = torch.sigmoid(self.semantic_gate(relation))
        semantic_delta = semantic_gate * (y_up - x)

        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        detail = self.detail_transform(x - local_mean)
        detail_gate = torch.sigmoid(self.detail_gate(relation))
        detail_delta = detail_gate * detail

        return x + self.semantic_gamma * semantic_delta + \
            self.detail_gamma * detail_delta


@MODELS.register_module()
class SegformerSGPCFHead(SegformerHead):
    """SegFormer head with SG-PCF C-v2 and no residual ASPP.

    This is the strict Module C implementation used by the A+C ablation.
    SG-PCF runs after each stage projection and before resize/concat fusion.
    """

    _FUSE_STAGE_PAIRS = {
        'c4_c3': (2, 3),
        'c3_c2': (1, 2),
        'c2_c1': (0, 1),
        'c4_c2': (1, 3),
    }

    def __init__(self,
                 sg_channels=256,
                 sg_mid_channels=128,
                 sg_init_gamma=0.0,
                 sg_fuse_stages=('c4_c3', 'c3_c2'),
                 sg_sim_type='spatial',
                 **kwargs):
        super().__init__(**kwargs)

        if sg_channels != self.channels:
            raise ValueError(
                f'sg_channels ({sg_channels}) must match decode channels '
                f'({self.channels}) because SG-PCF runs after projection.')
        if isinstance(sg_fuse_stages, str):
            sg_fuse_stages = (sg_fuse_stages, )
        if sg_fuse_stages is None:
            sg_fuse_stages = ()

        invalid_stages = [
            stage for stage in sg_fuse_stages
            if stage not in self._FUSE_STAGE_PAIRS
        ]
        if invalid_stages:
            valid = tuple(self._FUSE_STAGE_PAIRS)
            raise ValueError(
                f'Invalid sg_fuse_stages {invalid_stages}; valid values are '
                f'{valid}.')

        self.sg_channels = sg_channels
        self.sg_mid_channels = sg_mid_channels
        self.sg_init_gamma = sg_init_gamma
        self.sg_fuse_stages = tuple(sg_fuse_stages)
        self.sg_sim_type = sg_sim_type
        self.sg_fuses = nn.ModuleDict()
        for stage in self.sg_fuse_stages:
            self.sg_fuses[stage] = SemanticGuidedFuse(
                channels=sg_channels,
                mid_channels=sg_mid_channels,
                init_gamma=sg_init_gamma,
                sim_type=sg_sim_type,
                align_corners=self.align_corners,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feats = [conv(x) for conv, x in zip(self.convs, inputs)]
        for stage in self.sg_fuse_stages:
            high_idx, low_idx = self._FUSE_STAGE_PAIRS[stage]
            feats[high_idx] = self.sg_fuses[stage](
                feats[high_idx], feats[low_idx])

        outs = [
            resize(
                input=feat,
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners) for feat in feats
        ]
        out = self.fusion_conv(torch.cat(outs, dim=1))
        return self.cls_seg(out)


@MODELS.register_module()
class SegformerResidualASPPSGPCFHead(SegformerResidualASPPHead):
    """SegFormer Residual-ASPP head with SG-PCF before concat fusion.

    SG-PCF is inserted after each stage's 1x1 projection and before resizing
    all stages to the c1 resolution. The default stage list implements C-v2:

    p3' = SGFuse(p3, p4)
    p2' = SGFuse(p2, p3')
    p1 and p4 are kept unchanged.
    """

    _FUSE_STAGE_PAIRS = {
        'c4_c3': (2, 3),
        'c3_c2': (1, 2),
        'c2_c1': (0, 1),
        'c4_c2': (1, 3),
    }

    def __init__(self,
                 sg_channels=256,
                 sg_mid_channels=128,
                 sg_init_gamma=0.0,
                 sg_fuse_stages=('c4_c3', 'c3_c2'),
                 sg_sim_type='spatial',
                 **kwargs):
        super().__init__(**kwargs)

        if sg_channels != self.channels:
            raise ValueError(
                f'sg_channels ({sg_channels}) must match decode channels '
                f'({self.channels}) because SG-PCF runs after projection.')
        if isinstance(sg_fuse_stages, str):
            sg_fuse_stages = (sg_fuse_stages, )
        if sg_fuse_stages is None:
            sg_fuse_stages = ()

        self.sg_channels = sg_channels
        self.sg_mid_channels = sg_mid_channels
        self.sg_init_gamma = sg_init_gamma
        self.sg_fuse_stages = tuple(sg_fuse_stages)
        self.sg_sim_type = sg_sim_type

        invalid_stages = [
            stage for stage in self.sg_fuse_stages
            if stage not in self._FUSE_STAGE_PAIRS
        ]
        if invalid_stages:
            valid = tuple(self._FUSE_STAGE_PAIRS)
            raise ValueError(
                f'Invalid sg_fuse_stages {invalid_stages}; valid values are '
                f'{valid}.')

        self.sg_fuses = nn.ModuleDict()
        for stage in self.sg_fuse_stages:
            self.sg_fuses[stage] = SemanticGuidedFuse(
                channels=sg_channels,
                mid_channels=sg_mid_channels,
                init_gamma=sg_init_gamma,
                sim_type=sg_sim_type,
                align_corners=self.align_corners,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _apply_sg_pcf(self, feats):
        """Apply configured progressive cross-scale fusion in stage order."""
        for stage in self.sg_fuse_stages:
            high_idx, low_idx = self._FUSE_STAGE_PAIRS[stage]
            feats[high_idx] = self.sg_fuses[stage](
                feats[high_idx], feats[low_idx])
        return feats

    def forward(self, inputs):
        # Receive 4 stage backbone feature maps: c1, c2, c3, c4.
        inputs = self._transform_inputs(inputs)
        feats = [conv(x) for conv, x in zip(self.convs, inputs)]

        # SG-PCF runs after projection and before c1-scale resize/concat.
        feats = self._apply_sg_pcf(feats)

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
        out = self.cls_seg(out)
        return out


@MODELS.register_module()
class SegformerResidualASPPBPSFHead(SegformerResidualASPPHead):
    """Residual-ASPP decoder with boundary-preserving semantic fusion.

    BPSF is a conservative C-module variant for the full A+B+C model. It is
    inserted before the standard resize/concat fusion, starts as an identity
    mapping by default, and progressively injects high-level semantics while
    preserving target-layer details.
    """

    _FUSE_STAGE_PAIRS = {
        'c4_c3': (2, 3),
        'c3_c2': (1, 2),
        'c2_c1': (0, 1),
        'c4_c2': (1, 3),
    }

    def __init__(self,
                 bpsf_channels=256,
                 bpsf_mid_channels=128,
                 bpsf_init_semantic_gamma=0.0,
                 bpsf_init_detail_gamma=0.0,
                 bpsf_fuse_stages=('c4_c3', 'c3_c2'),
                 bpsf_sim_type='spatial',
                 **kwargs):
        super().__init__(**kwargs)

        if bpsf_channels != self.channels:
            raise ValueError(
                f'bpsf_channels ({bpsf_channels}) must match decode channels '
                f'({self.channels}) because BPSF runs after projection.')
        if isinstance(bpsf_fuse_stages, str):
            bpsf_fuse_stages = (bpsf_fuse_stages, )
        if bpsf_fuse_stages is None:
            bpsf_fuse_stages = ()

        invalid_stages = [
            stage for stage in bpsf_fuse_stages
            if stage not in self._FUSE_STAGE_PAIRS
        ]
        if invalid_stages:
            valid = tuple(self._FUSE_STAGE_PAIRS)
            raise ValueError(
                f'Invalid bpsf_fuse_stages {invalid_stages}; valid values '
                f'are {valid}.')

        self.bpsf_fuse_stages = tuple(bpsf_fuse_stages)
        self.bpsf_fuses = nn.ModuleDict()
        for stage in self.bpsf_fuse_stages:
            self.bpsf_fuses[stage] = BoundaryPreservingSemanticFuse(
                channels=bpsf_channels,
                mid_channels=bpsf_mid_channels,
                init_semantic_gamma=bpsf_init_semantic_gamma,
                init_detail_gamma=bpsf_init_detail_gamma,
                sim_type=bpsf_sim_type,
                align_corners=self.align_corners,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _apply_bpsf(self, feats):
        for stage in self.bpsf_fuse_stages:
            target_idx, guidance_idx = self._FUSE_STAGE_PAIRS[stage]
            feats[target_idx] = self.bpsf_fuses[stage](
                feats[target_idx], feats[guidance_idx])
        return feats

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feats = [conv(x) for conv, x in zip(self.convs, inputs)]
        feats = self._apply_bpsf(feats)

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
