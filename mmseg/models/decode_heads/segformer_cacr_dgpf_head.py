# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .segformer_head import SegformerHead
from .segformer_residual_aspp_head import SegformerResidualASPPHead


class ContentAdaptiveContextRefinement(nn.Module):
    """Content-adaptive multi-scale refinement for fused decoder features.

    Unlike ASPP's fixed branch concatenation, this block predicts per-image
    scale weights from global semantics and local contrast. A spatial gate then
    controls where the selected context is written back to the decoder feature.
    """

    def __init__(self,
                 channels=256,
                 branch_channels=64,
                 dilations=(1, 3, 6, 9),
                 selector_reduction=4,
                 init_scale=1e-3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        if not dilations:
            raise ValueError('dilations must contain at least one value.')

        self.channels = channels
        self.branch_channels = branch_channels
        self.dilations = tuple(dilations)

        self.context_branches = nn.ModuleList()
        for dilation in self.dilations:
            self.context_branches.append(
                nn.Sequential(
                    ConvModule(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    ConvModule(
                        channels,
                        branch_channels,
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)))

        # The global branch omits normalization because its feature map is 1x1.
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                channels,
                branch_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=act_cfg))

        self.num_branches = len(self.context_branches) + 1
        selector_channels = max(channels // selector_reduction, 16)
        self.scale_selector = nn.Sequential(
            nn.Conv2d(channels * 2, selector_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(selector_channels, self.num_branches, kernel_size=1))

        self.context_project = ConvModule(
            branch_channels,
            channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.spatial_selector = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True), nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid())

        # A small channel-wise scale keeps initialization close to identity
        # while allowing gradients to reach every new branch immediately.
        self.residual_scale = nn.Parameter(
            torch.full((channels, ), float(init_scale)))

    def _scale_weights(self, x):
        global_semantics = F.adaptive_avg_pool2d(x, 1)
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_contrast = F.adaptive_avg_pool2d(torch.abs(x - local_mean), 1)
        descriptor = torch.cat([global_semantics, local_contrast], dim=1)
        return torch.softmax(self.scale_selector(descriptor), dim=1)

    def forward(self, x, align_corners=False):
        branch_features = [branch(x) for branch in self.context_branches]
        global_context = resize(
            input=self.global_branch(x),
            size=x.shape[2:],
            mode='bilinear',
            align_corners=align_corners)
        branch_features.append(global_context)

        weights = self._scale_weights(x)
        stacked = torch.stack(branch_features, dim=1)
        context = torch.sum(stacked * weights.unsqueeze(2), dim=1)
        context = self.context_project(context)
        spatial_gate = self.spatial_selector(torch.cat([x, context], dim=1))
        update = spatial_gate * context
        scale = self.residual_scale.view(1, -1, 1, 1)
        return F.relu(x + scale * update, inplace=True)


class DifferenceAwareProgressiveFuse(nn.Module):
    """Dual-gate cross-scale fusion using semantic difference and detail.

    The unit does not blend two inputs with one convex gate. It predicts
    separate semantic-injection and detail-preservation gates from a relation
    tensor containing the target, guidance, absolute difference, and agreement
    features. The gated residuals are then written back with a small channel
    scale.
    """

    def __init__(self,
                 channels=256,
                 relation_channels=128,
                 init_scale=1e-3,
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.align_corners = align_corners
        self.target_project = ConvModule(
            channels,
            relation_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.guidance_project = ConvModule(
            channels,
            relation_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.relation_encoder = nn.Sequential(
            ConvModule(
                relation_channels * 4,
                relation_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                relation_channels,
                relation_channels,
                kernel_size=3,
                padding=1,
                groups=relation_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.gate_predictor = nn.Conv2d(relation_channels, 2, kernel_size=1)

        self.semantic_transform = nn.Sequential(
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
        self.detail_transform = nn.Sequential(
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.Conv2d(channels, channels, kernel_size=1))
        self.residual_scale = nn.Parameter(
            torch.full((channels, ), float(init_scale)))

    def forward(self, target, guidance):
        guidance = resize(
            input=guidance,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        target_relation = self.target_project(target)
        guidance_relation = self.guidance_project(guidance)
        relation = torch.cat([
            target_relation,
            guidance_relation,
            torch.abs(target_relation - guidance_relation),
            target_relation * guidance_relation,
        ],
                             dim=1)
        relation = self.relation_encoder(relation)
        semantic_gate, detail_gate = torch.sigmoid(
            self.gate_predictor(relation)).chunk(
                2, dim=1)

        semantic_residual = self.semantic_transform(guidance - target)
        local_mean = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        detail_residual = self.detail_transform(target - local_mean)
        update = semantic_gate * semantic_residual + \
            detail_gate * detail_residual
        scale = self.residual_scale.view(1, -1, 1, 1)
        return target + scale * update


@MODELS.register_module()
class SegformerCACRHead(SegformerHead):
    """SegFormer head with content-adaptive context refinement (CACR)."""

    def __init__(self,
                 context_branch_channels=64,
                 context_dilations=(1, 3, 6, 9),
                 context_selector_reduction=4,
                 context_init_scale=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.context_refinement = ContentAdaptiveContextRefinement(
            channels=self.channels,
            branch_channels=context_branch_channels,
            dilations=context_dilations,
            selector_reduction=context_selector_reduction,
            init_scale=context_init_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _project_and_resize(self, inputs):
        projected = [conv(x) for conv, x in zip(self.convs, inputs)]
        return [
            resize(
                input=feat,
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners) for feat in projected
        ]

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = self._project_and_resize(inputs)
        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.context_refinement(out, align_corners=self.align_corners)
        return self.cls_seg(out)


class _DGPFMixin:
    _FUSE_STAGE_PAIRS = {
        'c4_c3': (2, 3),
        'c3_c2': (1, 2),
        'c2_c1': (0, 1),
    }

    def _init_dgpf(self, relation_channels, init_scale, fuse_stages):
        if isinstance(fuse_stages, str):
            fuse_stages = (fuse_stages, )
        fuse_stages = tuple(fuse_stages or ())
        invalid = [
            stage for stage in fuse_stages
            if stage not in self._FUSE_STAGE_PAIRS
        ]
        if invalid:
            raise ValueError(
                f'Invalid dg_fuse_stages {invalid}; valid values are '
                f'{tuple(self._FUSE_STAGE_PAIRS)}.')

        self.dg_fuse_stages = fuse_stages
        self.dg_fuses = nn.ModuleDict()
        for stage in fuse_stages:
            self.dg_fuses[stage] = DifferenceAwareProgressiveFuse(
                channels=self.channels,
                relation_channels=relation_channels,
                init_scale=init_scale,
                align_corners=self.align_corners,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _apply_dgpf(self, feats):
        for stage in self.dg_fuse_stages:
            target_idx, guidance_idx = self._FUSE_STAGE_PAIRS[stage]
            feats[target_idx] = self.dg_fuses[stage](feats[target_idx],
                                                     feats[guidance_idx])
        return feats


@MODELS.register_module()
class SegformerDGPFHead(_DGPFMixin, SegformerHead):
    """SegFormer head with difference-aware progressive fusion (DGPF)."""

    def __init__(self,
                 dg_relation_channels=128,
                 dg_init_scale=1e-3,
                 dg_fuse_stages=('c4_c3', 'c3_c2'),
                 **kwargs):
        super().__init__(**kwargs)
        self._init_dgpf(dg_relation_channels, dg_init_scale, dg_fuse_stages)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feats = [conv(x) for conv, x in zip(self.convs, inputs)]
        feats = self._apply_dgpf(feats)
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
class SegformerResidualASPPDGPFHead(_DGPFMixin, SegformerResidualASPPHead):
    """Residual-ASPP decoder with DGPF inserted before multi-scale fusion.

    This keeps Module B as the original residual ASPP block and only replaces
    Module C with the difference-aware progressive fusion unit.
    """

    def __init__(self,
                 dg_relation_channels=128,
                 dg_init_scale=1e-3,
                 dg_fuse_stages=('c4_c3', 'c3_c2'),
                 **kwargs):
        super().__init__(**kwargs)
        self._init_dgpf(dg_relation_channels, dg_init_scale, dg_fuse_stages)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feats = [conv(x) for conv, x in zip(self.convs, inputs)]
        feats = self._apply_dgpf(feats)
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


@MODELS.register_module()
class SegformerCACRDGPFHead(_DGPFMixin, SegformerCACRHead):
    """SegFormer decoder combining DGPF before fusion and CACR after it."""

    def __init__(self,
                 dg_relation_channels=128,
                 dg_init_scale=1e-3,
                 dg_fuse_stages=('c4_c3', 'c3_c2'),
                 **kwargs):
        super().__init__(**kwargs)
        self._init_dgpf(dg_relation_channels, dg_init_scale, dg_fuse_stages)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feats = [conv(x) for conv, x in zip(self.convs, inputs)]
        feats = self._apply_dgpf(feats)
        outs = [
            resize(
                input=feat,
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners) for feat in feats
        ]
        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.context_refinement(out, align_corners=self.align_corners)
        return self.cls_seg(out)
