import numpy as np
from PIL import Image

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadBoundaryMask(BaseTransform):

    def transform(self, results):

        # segmentation GT 路径
        seg_map_path = results['seg_map_path']

        # 替换成 boundary 路径
        boundary_path = seg_map_path.replace(
            'Segmentation',
            'boundary'
        )

        # 读取 boundary mask
        boundary = np.array(Image.open(boundary_path))

        # 归一化
        boundary = boundary.astype(np.float32) / 255.0

        # 保存进 results
        results['gt_boundary_map'] = boundary
        results.setdefault('seg_fields', []).append('gt_boundary_map')

        return results
