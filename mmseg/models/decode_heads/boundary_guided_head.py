import torch
import torch.nn as nn

from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.registry import MODELS


@MODELS.register_module()
class BoundaryGuidedSegformerHead(SegformerHead):
    """SegFormerHead with boundary feature feedback.

    Boundary features from the auxiliary boundary head are used to generate
    a spatial attention map that guides the decoder's fused features,
    improving boundary-aware segmentation.

    Pipeline:
        backbone feats → conv1x1 + upsample → concat → fusion_conv
            ↓
        boundary_feat (64ch) → Conv3x3→1ch→Sigmoid → spatial gate
            ↓
        fused × (1 + gate) → cls_seg → seg_logits
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # spatial gate: map boundary feat (channels) → 1-ch spatial attention
        # but boundary feat is from separate module, we adapt here
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(64, self.channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs, boundary_feat=None):
        # Fallback to stored attribute (from encoder_decoder)
        if boundary_feat is None and hasattr(self, '_boundary_feat'):
            boundary_feat = self._boundary_feat

        # Standard SegFormerHead processing
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                self._resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                ))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        # Boundary feedback: spatial attention gate
        if boundary_feat is not None:
            gate = self.boundary_gate(boundary_feat)
            out = out * (1 + gate)

        out = self.cls_seg(out)
        return out

    def _resize(self, input, size):
        from mmseg.models.utils import resize
        return resize(
            input=input, size=size, mode='bilinear', align_corners=self.align_corners)
