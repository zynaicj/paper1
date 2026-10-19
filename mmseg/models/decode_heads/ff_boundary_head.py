import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils.freq_fusion import FreqFusion


@MODELS.register_module()
class FFBoundaryHead(BaseDecodeHead):
    """Boundary Head with FreqFusion (CVPR 2024).

    Two fusion modes:
      - 'concat': cat(c1_enhanced, cascade_up) → Conv(2C→C)  (richer info)
      - 'add':    c1_enhanced + cascade_up                     (lighter)

    Args:
        fusion_mode (str): 'concat' or 'add'.
    """

    def __init__(self, fusion_mode='concat', **kwargs):
        super().__init__(
            input_transform='multiple_select',
            **kwargs
        )
        assert fusion_mode in ('concat', 'add')
        self.fusion_mode = fusion_mode

        C = self.channels
        N = len(self.in_channels)

        # 1x1 通道统一
        self.ch_adapters = nn.ModuleList([
            nn.Conv2d(ic, C, 1) for ic in self.in_channels
        ])

        # 原版 FreqFusion cascade:
        # forward 从低到高迭代: k = N-2, N-3, ..., 0
        # ─ k=N-2 (最低): lr=feats[-1], 1 个特征 → lr_channels = C
        # ─ k=N-3:        lr=cat(feats[N-2]', up), 2 个特征 → lr_channels = 2C
        # ─ k=0 (最高):    lr=cat(feats[1]', up), N-1 个特征 → lr_channels = (N-1)*C
        self.ff_modules = nn.ModuleList()
        for k in range(N - 1):
            lr_ch = (N - 1 - k) * C
            self.ff_modules.append(
                FreqFusion(hr_channels=C, lr_channels=lr_ch,
                           compressed_channels=C)
            )

        # concat 模式最终 conv: 所有级联输出 + c1 = N * C 通道
        if self.fusion_mode == 'concat':
            self.final_conv = ConvModule(
                in_channels=N * C, out_channels=C,
                kernel_size=3, padding=1
            )
        else:
            self.final_conv = nn.Identity()

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        # 1. 通道统一: 从高分辨率到低分辨率排列
        feats = [adapter(inp) for adapter, inp in zip(self.ch_adapters, inputs)]
        # feats: [x1@H, x2@H/2, ..., xN@H/2^(N-1)]

        # 2. 原版 cascade: 最低层开始，逐级 FreqFusion + concat
        lr_accum = feats[-1]       # 最低分辨率
        cascade_ups = []           # 每层上采样后的结果

        # k=0..N-2, 对应 (cN-1,cN), (cN-2,cat(cN-1,cN_up)), ...
        for k in range(len(feats) - 2, -1, -1):
            hr = feats[k]
            ff = self.ff_modules[k]
            _, hr_out, lr_up = ff(hr_feat=hr, lr_feat=lr_accum)
            feats[k] = hr_out
            cascade_ups.append(lr_up)
            # 下一级 lr = cat(当前层 hr_out, 上采样结果)
            lr_accum = torch.cat([hr_out, lr_up], dim=1)

        # 3. 融合: c1 + 所有 cascade 上采样特征
        all_feats = [feats[0]] + cascade_ups
        if self.fusion_mode == 'concat':
            x = torch.cat(all_feats, dim=1)
            x = self.final_conv(x)
        else:
            x = sum(all_feats)

        x = self.cls_seg(x)
        return x

    def loss_by_feat(self, seg_logits, batch_data_samples):
        gt_boundary = []
        for data_sample in batch_data_samples:
            boundary = data_sample.gt_boundary_map.data
            gt_boundary.append(boundary)

        gt_boundary = torch.stack(gt_boundary).float()

        seg_logits = F.interpolate(
            seg_logits,
            size=gt_boundary.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        loss_boundary = self.loss_decode(
            seg_logits,
            gt_boundary.squeeze(1)
        )

        return dict(loss_boundary=loss_boundary)
