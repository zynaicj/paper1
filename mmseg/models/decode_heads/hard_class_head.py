import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class HardClassSemanticHead(BaseDecodeHead):
    """Binary auxiliary head for a target hard semantic class.

    This head reads deep backbone features and predicts whether each pixel
    belongs to ``target_class_id``. It is intended for training-time auxiliary
    supervision only.
    """

    def __init__(self,
                 target_class_id=2,
                 loss_weight=0.1,
                 bce_weight=1.0,
                 dice_weight=0.0,
                 pos_weight=None,
                 dice_smooth=1.0,
                 **kwargs):
        super().__init__(
            input_transform='multiple_select',
            **kwargs)
        self.target_class_id = target_class_id
        self.loss_weight = loss_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.dice_smooth = dice_smooth

        self.align_c3 = nn.Conv2d(self.in_channels[0], self.channels, 1)
        self.align_c4 = nn.Conv2d(self.in_channels[1], self.channels, 1)
        self.conv = ConvModule(
            self.channels * 2,
            self.channels,
            3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))
        self.conv_seg = nn.Conv2d(self.channels, 1, kernel_size=1)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        c3, c4 = inputs

        c4 = F.interpolate(
            c4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        c3 = self.align_c3(c3)
        c4 = self.align_c4(c4)

        x = torch.cat([c3, c4], dim=1)
        x = self.conv(x)
        return self.conv_seg(x)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        gt_seg_list = []
        for data_sample in batch_data_samples:
            gt_seg = data_sample.gt_sem_seg.data
            if gt_seg.dim() == 3 and gt_seg.size(0) == 1:
                gt_seg = gt_seg.squeeze(0)
            gt_seg_list.append(gt_seg.long())

        gt_seg = torch.stack(gt_seg_list, dim=0)
        seg_logits = F.interpolate(
            seg_logits,
            size=gt_seg.shape[-2:],
            mode='bilinear',
            align_corners=False)
        seg_logits = seg_logits.squeeze(1)

        valid_mask = gt_seg != self.ignore_index
        if not valid_mask.any():
            return dict(loss_hard_class=seg_logits.sum() * 0)

        hard_gt = (gt_seg == self.target_class_id).float()
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = seg_logits.new_tensor(self.pos_weight)
        loss_bce = F.binary_cross_entropy_with_logits(
            seg_logits, hard_gt, reduction='none', pos_weight=pos_weight)
        loss_bce = loss_bce[valid_mask].mean()

        loss = self.bce_weight * loss_bce
        if self.dice_weight > 0 and hard_gt[valid_mask].sum() > 0:
            pred = seg_logits.sigmoid()[valid_mask]
            target = hard_gt[valid_mask]
            intersection = (pred * target).sum()
            denominator = pred.sum() + target.sum()
            loss_dice = 1 - (2 * intersection + self.dice_smooth) / (
                denominator + self.dice_smooth)
            loss = loss + self.dice_weight * loss_dice

        return dict(loss_hard_class=loss * self.loss_weight)
