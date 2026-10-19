# Haar Wavelet Downsampling (HWD)
# Paper: https://arxiv.org/abs/2111.11336
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class HaarDownsampling(nn.Module):
    """Haar Wavelet-based Downsampling.

    Lossless spatial-to-channel encoding: decomposes H×W into 4 components
    at H/2×W/2, then 1x1 conv for channel alignment.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)
        out_size = x.shape[-2:]
        # PatchEmbed 兼容格式: [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        return x, out_size
