import numpy as np
import torch

from mmcv.transforms import BaseTransform
from mmcv.transforms.formatting import to_tensor

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData


@TRANSFORMS.register_module()
class PackBoundaryInputs(BaseTransform):

    def transform(self, results):

        packed_results = dict()

        # image
        if 'img' in results:
            img = results['img']

            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)

            img = np.ascontiguousarray(img.transpose(2, 0, 1))

            packed_results['inputs'] = to_tensor(img)

        data_sample = SegDataSample()

        # segmentation GT
        if 'gt_seg_map' in results:

            gt_sem_seg_data = dict(
                data=to_tensor(
                    results['gt_seg_map'][None, ...].astype(np.int64)
                )
            )

            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        # boundary GT
        if 'gt_boundary_map' in results:

            gt_boundary_data = dict(
                data=to_tensor(
                    results['gt_boundary_map'][None, ...].astype(np.float32)
                )
            )

            data_sample.set_data({
                'gt_boundary_map':
                PixelData(**gt_boundary_data)
            })

        packed_results['data_samples'] = data_sample

        return packed_results
