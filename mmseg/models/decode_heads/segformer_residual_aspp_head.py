# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from .aspp_head import ASPPModule
from .upcm import UPCM


class ASPPContextBlock(nn.Module):
    """ASPP context block for a fused SegFormer decoder feature."""

    def __init__(self,
                 in_channels,
                 aspp_channels=128,
                 dilations=(1, 6, 12, 18),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels,
                aspp_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            in_channels,
            aspp_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * aspp_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x, align_corners=False):
        aspp_outs = [
            resize(
                input=self.image_pool(x),
                size=x.shape[2:],
                mode='bilinear',
                align_corners=align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        return self.bottleneck(torch.cat(aspp_outs, dim=1))


class GlobalOnlyContextBlock(nn.Module):
    """Image-level context branch without regional ASPP branches."""

    def __init__(self,
                 in_channels,
                 context_channels=128,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels,
                context_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.bottleneck = ConvModule(
            context_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x, align_corners=False):
        context = resize(
            input=self.image_pool(x),
            size=x.shape[2:],
            mode='bilinear',
            align_corners=align_corners)
        return self.bottleneck(context)


class RegionalOnlyContextBlock(nn.Module):
    """Multi-receptive-field regional context without image pooling."""

    def __init__(self,
                 in_channels,
                 aspp_channels=128,
                 dilations=(1, 6, 12, 18),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.aspp_modules = ASPPModule(
            dilations,
            in_channels,
            aspp_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            len(dilations) * aspp_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x, align_corners=False):
        return self.bottleneck(torch.cat(self.aspp_modules(x), dim=1))


class ResidualASPP(nn.Module):
    """Residual ASPP wrapper that starts close to an identity mapping."""

    def __init__(self,
                 channels,
                 aspp_channels=128,
                 dilations=(1, 6, 12, 18),
                 init_gamma=0.0,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.aspp = ASPPContextBlock(
            channels,
            aspp_channels=aspp_channels,
            dilations=dilations,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x, align_corners=False):
        residual = self.aspp(x, align_corners=align_corners)
        return F.relu(x + self.gamma * residual, inplace=True)


class ResidualGlobalOnly(nn.Module):
    """Residual wrapper for image-level context only."""

    def __init__(self,
                 channels,
                 context_channels=128,
                 init_gamma=0.0,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.context = GlobalOnlyContextBlock(
            channels,
            context_channels=context_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x, align_corners=False):
        residual = self.context(x, align_corners=align_corners)
        return F.relu(x + self.gamma * residual, inplace=True)


class ResidualRegionalOnly(nn.Module):
    """Residual wrapper for regional ASPP context only."""

    def __init__(self,
                 channels,
                 aspp_channels=128,
                 dilations=(1, 6, 12, 18),
                 init_gamma=0.0,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.context = RegionalOnlyContextBlock(
            channels,
            aspp_channels=aspp_channels,
            dilations=dilations,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x, align_corners=False):
        residual = self.context(x, align_corners=align_corners)
        return F.relu(x + self.gamma * residual, inplace=True)


class ConvMlp(nn.Module):
    """1x1 convolutional MLP that keeps spatial resolution."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 norm_layer=None,
                 bias=True,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(
            in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RCA(nn.Module):
    """Rectangular self-calibration attention."""

    def __init__(self,
                 inp,
                 kernel_size=1,
                 ratio=1,
                 band_kernel_size=11,
                 dw_size=(1, 1),
                 padding=(0, 0),
                 stride=1,
                 square_kernel_size=3,
                 relu=True):
        super().__init__()
        self.dwconv_hw = nn.Conv2d(
            inp,
            inp,
            square_kernel_size,
            padding=square_kernel_size // 2,
            groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(
                inp,
                gc,
                kernel_size=(1, band_kernel_size),
                padding=(0, band_kernel_size // 2),
                groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                gc,
                inp,
                kernel_size=(band_kernel_size, 1),
                padding=(band_kernel_size // 2, 0),
                groups=gc),
            nn.Sigmoid())

    def sge(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w
        return self.excite(x_gather)

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        return att * loc


class RCM(nn.Module):
    """Rectangular self-calibration module."""

    def __init__(self,
                 dim,
                 token_mixer=RCA,
                 norm_layer=nn.BatchNorm2d,
                 mlp_layer=ConvMlp,
                 mlp_ratio=2,
                 act_layer=nn.GELU,
                 ls_init_value=1e-6,
                 drop_path=0.,
                 dw_size=11,
                 square_kernel_size=3,
                 ratio=1):
        super().__init__()
        self.token_mixer = token_mixer(
            dim,
            band_kernel_size=dw_size,
            square_kernel_size=square_kernel_size,
            ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) \
            if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return self.drop_path(x) + shortcut


class LSKblock(nn.Module):
    """Large selective kernel spatial attention block."""

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
            attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


@MODELS.register_module()
class SegformerResidualASPPHead(BaseDecodeHead):
    """SegFormer head with residual ASPP after multi-scale fusion.

    The original SegFormerHead four-scale projection, resize, concat, and
    fusion path is preserved. ASPP is inserted after the fused feature and
    before ``cls_seg`` to provide multi-scale regional context that can
    complement BoundaryHead supervision.
    """

    def __init__(self,
                 interpolate_mode='bilinear',
                 aspp_channels=128,
                 dilations=(1, 6, 12, 18),
                 aspp_init_gamma=0.0,
                 upcm_cfg=None,
                 use_rcm=False,
                 use_lsk=False,
                 rcm_cfg=None,
                 lsk_cfg=None,
                 rcm_init_gamma=0.0,
                 lsk_init_gamma=0.0,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        if use_rcm and use_lsk:
            raise ValueError('RCM and LSK are mutually exclusive.')
        if upcm_cfg is not None and (use_rcm or use_lsk):
            raise ValueError('UPCM cannot be enabled together with RCM or LSK.')

        self.interpolate_mode = interpolate_mode
        self.aspp_channels = aspp_channels
        self.dilations = dilations
        self.upcm_cfg = upcm_cfg
        self.use_rcm = use_rcm
        self.use_lsk = use_lsk
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

        self.residual_aspp = ResidualASPP(
            self.channels,
            aspp_channels=aspp_channels,
            dilations=dilations,
            init_gamma=aspp_init_gamma,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if upcm_cfg is not None:
            upcm_cfg = upcm_cfg.copy()
            self.upcm = UPCM(
                channels=self.channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                **upcm_cfg)
        else:
            self.upcm = None

        if use_rcm:
            rcm_cfg = rcm_cfg.copy() if rcm_cfg is not None else {}
            self.rcm = RCM(dim=self.channels, **rcm_cfg)
            self.gamma_rcm = nn.Parameter(torch.tensor(float(rcm_init_gamma)))
        else:
            self.rcm = None
            self.gamma_rcm = None

        if use_lsk:
            lsk_cfg = lsk_cfg.copy() if lsk_cfg is not None else {}
            self.lsk = LSKblock(dim=self.channels, **lsk_cfg)
            self.gamma_lsk = nn.Parameter(torch.tensor(float(lsk_init_gamma)))
        else:
            self.lsk = None
            self.gamma_lsk = None

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
class SegformerResidualGlobalOnlyHead(SegformerResidualASPPHead):
    """SegFormer head with residual image-level context only."""

    def __init__(self,
                 aspp_channels=128,
                 dilations=(1, 6, 12, 18),
                 aspp_init_gamma=0.0,
                 **kwargs):
        super().__init__(
            aspp_channels=aspp_channels,
            dilations=dilations,
            aspp_init_gamma=aspp_init_gamma,
            **kwargs)
        self.residual_aspp = ResidualGlobalOnly(
            self.channels,
            context_channels=aspp_channels,
            init_gamma=aspp_init_gamma,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


@MODELS.register_module()
class SegformerResidualRegionalOnlyHead(SegformerResidualASPPHead):
    """SegFormer head with residual multi-dilation regional context only."""

    def __init__(self,
                 aspp_channels=128,
                 dilations=(1, 6, 12, 18),
                 aspp_init_gamma=0.0,
                 **kwargs):
        super().__init__(
            aspp_channels=aspp_channels,
            dilations=dilations,
            aspp_init_gamma=aspp_init_gamma,
            **kwargs)
        self.residual_aspp = ResidualRegionalOnly(
            self.channels,
            aspp_channels=aspp_channels,
            dilations=dilations,
            init_gamma=aspp_init_gamma,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
