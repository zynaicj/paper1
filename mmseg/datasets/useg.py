from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class USEGDataset(BaseSegDataset):
    """USEG underwater semantic segmentation dataset.

    The semantic masks in ``Segmentation`` are single-channel indexed PNGs with
    labels 0-4. ``SegmentationClassVisual`` contains RGB visualization masks and
    should not be used for training.
    """

    METAINFO = dict(
        classes=(
            'background',
            'holothurian',
            'echinus',
            'scallop',
            'starfish',
        ),
        palette=[
            [0, 0, 0],        # background
            [255, 0, 0],      # holothurian
            [0, 255, 0],      # echinus
            [0, 0, 255],      # scallop
            [255, 255, 0],    # starfish
        ])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
