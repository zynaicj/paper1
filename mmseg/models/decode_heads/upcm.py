# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmseg.registry import MODELS


class SimAM(nn.Module):
    """Parameter-free SimAM attention."""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        n = h * w - 1
        if n <= 0:
            return x
        x_minus_mu_square = (
            x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=(2, 3), keepdim=True) / n
                 + self.e_lambda)) + 0.5
        return x * self.act(y)


class HSigmoid(nn.Module):

    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HSwish(nn.Module):

    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    """Coordinate attention with separate horizontal/vertical context."""

    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


@MODELS.register_module()
class UPCM(nn.Module):
    """Underwater Position-Contrast Modulation.

    UPCM recalibrates low-contrast underwater decoder features by interacting
    SimAM saliency responses and coordinate-aware positional features. The
    residual scale starts from zero so the module is initialized as identity.
    """

    def __init__(self,
                 channels,
                 reduction=32,
                 simam_e_lambda=1e-4,
                 use_simam=True,
                 use_coord=True,
                 use_gate=True,
                 init_gamma=0.0,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        if not use_simam and not use_coord:
            raise ValueError('UPCM needs at least one enabled branch.')
        self.channels = channels
        self.use_simam = use_simam
        self.use_coord = use_coord
        self.use_gate = use_gate

        self.simam = SimAM(simam_e_lambda) if use_simam else None
        self.coord = CoordAttention(
            channels, channels, reduction=reduction) if use_coord else None

        branch_channels = channels * int(use_simam) + channels * int(use_coord)
        self.interaction = ConvModule(
            branch_channels,
            channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if use_gate:
            self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        else:
            self.gate = None

        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        self.latest_gate_mean = None
        self.latest_gate_max = None

    def forward(self, x):
        branches = []
        if self.use_simam:
            branches.append(self.simam(x))
        if self.use_coord:
            branches.append(self.coord(x))

        if len(branches) == 1:
            fusion = branches[0]
        else:
            fusion = torch.cat(branches, dim=1)
        f_inter = self.interaction(fusion)

        if self.use_gate:
            gate = torch.sigmoid(self.gate(torch.cat([x, f_inter], dim=1)))
            f_mod = f_inter * gate
            if not self.training:
                self.latest_gate_mean = gate.detach().mean()
                self.latest_gate_max = gate.detach().max()
        else:
            f_mod = f_inter
            self.latest_gate_mean = None
            self.latest_gate_max = None

        return x + self.gamma * f_mod
