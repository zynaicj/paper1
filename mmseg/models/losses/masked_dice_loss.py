# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class MaskedDiceLoss(nn.Module):
    """Multi-class Dice loss with spatial ignore-mask support.

    This loss is intended for semantic segmentation logits of shape
    ``[N, C, H, W]`` and integer labels of shape ``[N, H, W]``. Pixels equal
    to ``ignore_index`` do not contribute to either the numerator or
    denominator.
    """

    def __init__(self,
                 use_sigmoid: bool = False,
                 activate: bool = True,
                 exclude_background: bool = False,
                 skip_empty_classes: bool = True,
                 class_weight: Optional[Sequence[float]] = None,
                 loss_weight: float = 1.0,
                 ignore_index: int = 255,
                 eps: float = 1.0,
                 loss_name: str = 'loss_dice'):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.exclude_background = exclude_background
        self.skip_empty_classes = skip_empty_classes
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.eps = eps
        self._loss_name = loss_name

        if class_weight is not None:
            self.register_buffer(
                'class_weight',
                torch.tensor(class_weight, dtype=torch.float32),
                persistent=False)
        else:
            self.class_weight = None

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                ignore_index: Optional[int] = None,
                **kwargs) -> torch.Tensor:
        del avg_factor, kwargs
        if reduction_override not in (None, 'mean'):
            raise ValueError('MaskedDiceLoss only supports mean reduction.')

        if target.ndim == 4:
            target = target.squeeze(1)
        target = target.long()

        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        valid_mask = target != ignore_index
        if not torch.any(valid_mask):
            return pred.sum() * 0

        if self.activate:
            if self.use_sigmoid:
                prob = pred.sigmoid()
            else:
                prob = pred.softmax(dim=1)
        else:
            prob = pred

        num_classes = pred.shape[1]
        safe_target = target.clone()
        safe_target[~valid_mask] = 0
        one_hot = F.one_hot(safe_target, num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=prob.dtype)

        valid_weight = valid_mask.unsqueeze(1).to(dtype=prob.dtype)
        if weight is not None:
            if weight.ndim == 3:
                weight = weight.unsqueeze(1)
            valid_weight = valid_weight * weight.to(dtype=prob.dtype)

        prob = prob * valid_weight
        one_hot = one_hot * valid_weight

        if self.exclude_background and num_classes > 1:
            prob = prob[:, 1:]
            one_hot = one_hot[:, 1:]
            class_offset = 1
        else:
            class_offset = 0

        dims = (0, 2, 3)
        intersection = (prob * one_hot).sum(dim=dims)
        denominator = prob.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        loss = 1 - dice

        valid_classes = torch.ones_like(loss, dtype=torch.bool)
        if self.skip_empty_classes:
            valid_classes = one_hot.sum(dim=dims) > 0

        if self.class_weight is not None:
            class_weight = self.class_weight.to(device=loss.device,
                                                dtype=loss.dtype)
            class_weight = class_weight[class_offset:class_offset + loss.numel()]
            valid_loss = loss[valid_classes] * class_weight[valid_classes]
            weight_sum = class_weight[valid_classes].sum().clamp_min(self.eps)
            dice_loss = valid_loss.sum() / weight_sum
        else:
            if not torch.any(valid_classes):
                return pred.sum() * 0
            dice_loss = loss[valid_classes].mean()

        return self.loss_weight * dice_loss

    @property
    def loss_name(self):
        return self._loss_name
