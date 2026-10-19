import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.models.utils.dynamic_filter import DynamicFilter


@MODELS.register_module()
class DynamicFilterNeck(nn.Module):
    """DynamicFilter feature refinement neck.

    FFT-based global token mixing applied to each backbone output.
    Shape-preserving.

    Args:
        in_channels (list[int]): Channel counts per feature level.
    """

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.df_modules = nn.ModuleList([
            DynamicFilter(dim=c, **kwargs)
            for c in in_channels
        ])

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        outputs = []
        for m, x in zip(self.df_modules, inputs):
            # BCHW → BHWC (DynamicFilter expects BHWC)
            x = x.permute(0, 2, 3, 1)
            x = m(x)
            x = x.permute(0, 3, 1, 2)  # BHWC → BCHW
            outputs.append(x)
        return outputs
