# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


IndexKey = Union[str, int]


@MODELS.register_module()
class UCRSLoss(nn.Module):
    """Underwater Confusion-aware Region-Scale Loss.

    UCRSLoss is designed for underwater semantic segmentation. It improves
    difficult-class supervision by enforcing region-scale consistency and
    suppressing semantic confusion among visually similar underwater
    categories.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 lambda_rs: float = 0.1,
                 lambda_cm: float = 0.03,
                 gamma: float = 2.0,
                 pool_scales: List[int] = [4, 8, 16, 32],
                 ignore_index: int = 255,
                 eps: float = 1e-6,
                 class_index: Dict[IndexKey, int] = None,
                 confusion_map: Dict[IndexKey, List[IndexKey]] = None,
                 loss_name: str = 'loss_ucrs'):
        super().__init__()
        self.loss_weight = loss_weight
        self.lambda_rs = lambda_rs
        self.lambda_cm = lambda_cm
        self.gamma = gamma
        self.pool_scales = tuple(pool_scales)
        self.ignore_index = ignore_index
        self.eps = eps
        self.class_index = class_index or {
            'aquatic_plants': 2,
            'robots': 4,
            'fish': 6,
            'sea_floor': 7,
        }
        self.confusion_map = self._parse_confusion_map(confusion_map or {
            'aquatic_plants': [7, 5, 0],
            'robots': [3, 0],
            'fish': [0, 5, 7],
            'sea_floor': [5, 2, 0],
        })
        self.difficult_classes = tuple(self._to_index(k)
                                       for k in self.class_index.keys())
        self._loss_name = loss_name

    def _to_index(self, key: IndexKey) -> int:
        if isinstance(key, str):
            if key not in self.class_index:
                raise KeyError(f'Unknown class name in UCRSLoss: {key}')
            return int(self.class_index[key])
        return int(key)

    def _parse_confusion_map(
            self, confusion_map: Dict[IndexKey, List[IndexKey]]
    ) -> Dict[int, List[int]]:
        parsed = {}
        for key, values in confusion_map.items():
            parsed[self._to_index(key)] = [self._to_index(v) for v in values]
        return parsed

    def _check_indices(self, num_classes: int) -> None:
        all_indices = set(self.difficult_classes)
        for values in self.confusion_map.values():
            all_indices.update(values)
        invalid = sorted(idx for idx in all_indices
                         if idx < 0 or idx >= num_classes)
        if invalid:
            raise ValueError(
                f'UCRSLoss got invalid class indices {invalid} for '
                f'prediction with {num_classes} classes.')

    def _region_scale_loss(self, prob: torch.Tensor, target: torch.Tensor,
                           valid_mask: torch.Tensor) -> torch.Tensor:
        """Match local predicted and GT region ratios for difficult classes."""
        _, _, height, width = prob.shape
        losses = []
        valid_mask_4d = valid_mask.unsqueeze(1).float()

        for class_idx in self.difficult_classes:
            gt_class = ((target == class_idx) & valid_mask).unsqueeze(1).float()
            if gt_class.sum() <= 0:
                continue

            pred_class = prob[:, class_idx:class_idx + 1] * valid_mask_4d
            gt_class = gt_class * valid_mask_4d

            for scale in self.pool_scales:
                if height < scale or width < scale:
                    continue

                valid_pool = F.avg_pool2d(
                    valid_mask_4d, kernel_size=scale, stride=scale)
                pred_pool = F.avg_pool2d(
                    pred_class, kernel_size=scale, stride=scale)
                gt_pool = F.avg_pool2d(
                    gt_class, kernel_size=scale, stride=scale)

                pred_ratio = pred_pool / valid_pool.clamp_min(self.eps)
                gt_ratio = gt_pool / valid_pool.clamp_min(self.eps)
                active_region = gt_pool > 0
                if active_region.sum() <= 0:
                    continue

                region_error = (pred_ratio - gt_ratio).abs()
                losses.append(region_error[active_region].mean())

        if not losses:
            return prob.sum() * 0.0
        return torch.stack(losses).mean()

    def _confusion_suppression_loss(self, prob: torch.Tensor,
                                    target: torch.Tensor,
                                    valid_mask: torch.Tensor) -> torch.Tensor:
        """Suppress confusing-class probability on difficult-class GT pixels."""
        total_loss = prob.sum() * 0.0
        total_count = prob.new_zeros(())

        for class_idx, confusing_indices in self.confusion_map.items():
            hard_mask = (target == class_idx) & valid_mask
            pixel_count = hard_mask.sum()
            if pixel_count <= 0:
                continue

            p_true = prob[:, class_idx][hard_mask]
            p_conf = prob[:, confusing_indices].sum(dim=1)[hard_mask]
            relative_prob = p_true / (p_true + p_conf + self.eps)
            focal_weight = (1.0 - p_true).clamp_min(0.0).pow(self.gamma)
            loss = -torch.log(relative_prob + self.eps) * focal_weight

            total_loss = total_loss + loss.sum()
            total_count = total_count + pixel_count.to(prob.dtype)

        if total_count <= 0:
            return prob.sum() * 0.0
        return total_loss / total_count.clamp_min(self.eps)

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
                f'UCRSLoss expects target shape [N,H,W] or [N,1,H,W], '
                f'but got {tuple(target.shape)}.')

        ignore_index = self.ignore_index if ignore_index is None \
            else ignore_index
        target = target.to(device=pred.device, dtype=torch.long)

        if pred.shape[2:] != target.shape[1:]:
            pred = F.interpolate(
                pred,
                size=target.shape[1:],
                mode='bilinear',
                align_corners=False)

        self._check_indices(pred.size(1))
        prob = pred.softmax(dim=1)
        valid_mask = target != ignore_index

        zero_loss = prob.sum() * 0.0
        loss_rs = self._region_scale_loss(prob, target, valid_mask) \
            if self.lambda_rs > 0 else zero_loss
        loss_cm = self._confusion_suppression_loss(prob, target, valid_mask) \
            if self.lambda_cm > 0 else zero_loss
        loss = self.lambda_rs * loss_rs + self.lambda_cm * loss_cm
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
