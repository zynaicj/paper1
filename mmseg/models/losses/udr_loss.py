# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class UDRLoss(nn.Module):
    """Uncertainty-guided Difficult Region Loss.

    UDRLoss is designed for underwater semantic segmentation. It adaptively
    increases supervision on uncertain pixels of difficult underwater
    categories, without enforcing hard area constraints or modifying the
    network architecture.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 lambda_udr: float = 0.1,
                 difficult_classes: Sequence[int] = (2, 4, 6, 7),
                 ignore_index: int = 255,
                 detach_uncertainty: bool = True,
                 min_weight: float = 1.0,
                 max_weight: float = 1.5,
                 normalize_weight: bool = False,
                 loss_name: str = 'loss_udr'):
        super().__init__()
        self.loss_weight = loss_weight
        self.lambda_udr = lambda_udr
        self.difficult_classes = tuple(int(c) for c in difficult_classes)
        self.ignore_index = ignore_index
        self.detach_uncertainty = detach_uncertainty
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.normalize_weight = normalize_weight
        self._loss_name = loss_name

    def _check_indices(self, num_classes: int) -> None:
        invalid = sorted(
            c for c in self.difficult_classes if c < 0 or c >= num_classes)
        if invalid:
            raise ValueError(
                f'UDRLoss got invalid difficult_classes {invalid} for '
                f'prediction with {num_classes} classes.')

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index: int = None,
                **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            pred (Tensor): Segmentation logits with shape [N, C, H, W].
            target (Tensor): Ground-truth labels with shape [N, H, W] or
                [N, 1, H, W].
        """
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        if target.dim() != 3:
            raise ValueError(
                f'UDRLoss expects target shape [N,H,W] or [N,1,H,W], '
                f'but got {tuple(target.shape)}.')

        ignore_index = self.ignore_index if ignore_index is None \
            else ignore_index
        target = target.to(device=pred.device, dtype=torch.long)

        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
                pred,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False)

        self._check_indices(pred.size(1))
        valid_mask = target != ignore_index
        valid_count = valid_mask.sum()
        if valid_count <= 0:
            return pred.sum() * 0.0

        ce_loss = F.cross_entropy(
            pred,
            target,
            reduction='none',
            ignore_index=ignore_index)

        prob = pred.softmax(dim=1)
        uncertainty = 1.0 - prob.max(dim=1).values
        if self.detach_uncertainty:
            uncertainty = uncertainty.detach()

        difficult_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        for class_idx in self.difficult_classes:
            difficult_mask = difficult_mask | (target == class_idx)
        difficult_mask = difficult_mask & valid_mask

        pixel_weight = 1.0 + self.lambda_udr * uncertainty * difficult_mask
        pixel_weight = pixel_weight.clamp(
            min=self.min_weight, max=self.max_weight)
        pixel_weight = pixel_weight * valid_mask.float()

        if weight is not None:
            weight = weight.to(device=pred.device, dtype=pred.dtype)
            if weight.dim() == 4 and weight.size(1) == 1:
                weight = weight.squeeze(1)
            pixel_weight = pixel_weight * weight

        if self.normalize_weight:
            avg_factor = pixel_weight.sum().clamp_min(1.0)
        else:
            avg_factor = valid_count.clamp_min(1)
        loss = (ce_loss * pixel_weight).sum() / avg_factor
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
