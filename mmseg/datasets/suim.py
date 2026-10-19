from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SUIMDataset(BaseSegDataset):

    METAINFO = dict(

        classes=(

            'background',

            'human_diver',

            'aquatic_plants',

            'wrecks_ruins',

            'robots',

            'reefs',

            'fish',

            'sea_floor'
        ),

        palette=[

            [0, 0, 0],        # background

            [255, 0, 0],      # human_diver

            [0, 255, 0],      # aquatic_plants

            [255, 255, 0],    # wrecks_ruins

            [0, 0, 255],      # robots

            [255, 0, 255],    # reefs

            [0, 255, 255],    # fish

            [128, 128, 128]   # sea_floor
        ]
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs):

        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
