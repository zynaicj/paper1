# Adapted from https://github.com/Linwei-Chen/FreqFusion (CVPR 2024)
# Paper: Frequency-aware Feature Fusion for Dense Image Prediction
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import carafe
from mmengine.model import normal_init, xavier_init, constant_init


def hamming2d(M, N):
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    return np.outer(hamming_x, hamming_y)


class FreqFusion(nn.Module):
    """Frequency-aware Feature Fusion.

    Replaces bilinear upsampling with frequency-aware upsampling that
    preserves high-frequency boundary details.

    Args:
        hr_channels (int): High-resolution feature channels.
        lr_channels (int): Low-resolution feature channels.
        compressed_channels (int): Internal compressed channel count.
        lowpass_kernel (int): Low-pass filter kernel size.
        highpass_kernel (int): High-pass filter kernel size.
    """

    def __init__(self,
                 hr_channels,
                 lr_channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 compressed_channels=64,
                 align_corners=False,
                 upsample_mode='nearest',
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 semi_conv=True,
                 **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.compressed_channels = compressed_channels
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv

        encoder_dilation = 1
        self.hr_channel_compressor = nn.Conv2d(hr_channels, compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, compressed_channels, 1)

        self.content_encoder = nn.Conv2d(
            compressed_channels,
            lowpass_kernel ** 2 * up_group,
            encoder_kernel,
            padding=int((encoder_kernel - 1) * encoder_dilation / 2),
            dilation=encoder_dilation,
            groups=1)

        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(
                compressed_channels,
                highpass_kernel ** 2 * up_group,
                encoder_kernel,
                padding=int((encoder_kernel - 1) * encoder_dilation / 2),
                dilation=encoder_dilation,
                groups=1)

        self.register_buffer('hamming_lowpass',
                             torch.FloatTensor(hamming2d(lowpass_kernel, lowpass_kernel))[None, None])
        self.register_buffer('hamming_highpass',
                             torch.FloatTensor(hamming2d(highpass_kernel, highpass_kernel))[None, None])
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, hamming=1):
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))
        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, hr_feat, lr_feat):
        return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)

        if self.semi_conv and self.use_high_pass:
            mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
            mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel,
                                                   hamming=self.hamming_highpass)
            compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - \
                carafe(compressed_hr_feat, mask_hr_init, self.highpass_kernel, self.up_group, 1)

            mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
            mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel,
                                                   hamming=self.hamming_lowpass)

            mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
            mask_lr_lr_feat = F.interpolate(
                carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2),
                size=compressed_hr_feat.shape[-2:], mode='nearest')
            mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

            mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel,
                                                   hamming=self.hamming_lowpass)
            mask_hr_lr_feat = F.interpolate(
                carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init,
                       self.lowpass_kernel, self.up_group, 2),
                size=compressed_hr_feat.shape[-2:], mode='nearest')
            mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
        else:
            mask_lr = self.content_encoder(compressed_hr_feat) + \
                F.interpolate(self.content_encoder(compressed_lr_feat),
                              size=compressed_hr_feat.shape[-2:], mode='nearest')
            if self.use_high_pass:
                mask_hr = self.content_encoder2(compressed_hr_feat) + \
                    F.interpolate(self.content_encoder2(compressed_lr_feat),
                                  size=compressed_hr_feat.shape[-2:], mode='nearest')

        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel,
                                          hamming=self.hamming_lowpass)
        lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel,
                                              hamming=self.hamming_highpass)
            hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel,
                                           self.up_group, 1)
            hr_feat = hr_feat_hf + hr_feat

        return mask_lr, hr_feat, lr_feat
