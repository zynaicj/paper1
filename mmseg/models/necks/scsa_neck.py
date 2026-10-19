import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.models.utils.scsa import SCSA


@MODELS.register_module()
class SCSANeck(nn.Module):
    """SCSA feature refinement neck.

    Applies SCSA attention to each backbone output independently.
    Shape-preserving, inserted between backbone and decode/auxiliary heads.

    Args:
        in_channels (list[int]): Channel counts per feature level.
        head_num (int): Attention heads per SCSA module.
        window_size (int): Window size for channel attention downsampling.
    """

    def __init__(self,
                 in_channels,
                 head_num=4,
                 window_size=7):
        super().__init__()
        if isinstance(window_size, int):
            window_size = [window_size] * len(in_channels)
        self.scsa_modules = nn.ModuleList([
            SCSA(dim=c, head_num=head_num, window_size=w)
            for c, w in zip(in_channels, window_size)
        ])

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        return [m(x) for m, x in zip(self.scsa_modules, inputs)]
