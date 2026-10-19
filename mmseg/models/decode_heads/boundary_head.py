import torch
import torch.nn.functional as F

from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@MODELS.register_module()
class BoundaryHead(BaseDecodeHead):

    def __init__(self, **kwargs):

        super().__init__(
            input_transform='multiple_select',
            **kwargs
        )

        self.conv = ConvModule(
            in_channels=sum(self.in_channels),
            out_channels=64,
            kernel_size=3,
            padding=1
        )

    def _fuse_feats(self, inputs):
        """Fuse backbone features and return 64ch boundary feature."""
        inputs = self._transform_inputs(inputs)
        c1 = inputs[0]
        feats = [c1]
        for ci in inputs[1:]:
            ci = F.interpolate(
                ci,
                size=c1.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            feats.append(ci)
        x = torch.cat(feats, dim=1)
        return self.conv(x)

    def forward_boundary_feat(self, inputs):
        """Return boundary features for decoder guidance (64ch, H/4, W/4)."""
        return self._fuse_feats(inputs)

    def forward(self, inputs):
        x = self._fuse_feats(inputs)
        x = self.cls_seg(x)
        return x

    def loss_by_feat(self, seg_logits, batch_data_samples):

        gt_boundary = []
        target_size = batch_data_samples[0].gt_sem_seg.data.shape[-2:]

        for data_sample in batch_data_samples:

            boundary = data_sample.gt_boundary_map.data.float()
            if boundary.dim() == 2:
                boundary = boundary.unsqueeze(0)
            boundary = boundary.to(seg_logits.device)
            if boundary.shape[-2:] != target_size:
                boundary = F.interpolate(
                    boundary.unsqueeze(0),
                    size=target_size,
                    mode='nearest').squeeze(0)

            gt_boundary.append(boundary)

        gt_boundary = torch.stack(gt_boundary).float()

        # resize pred
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
