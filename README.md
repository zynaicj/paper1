# Structure-Aware Network for Underwater Image Semantic Segmentation

## Abstract

Color attenuation, uneven illumination, and complex backgrounds in underwater images lead to blurred object boundaries, substantial scale variation, and class confusion. To address these challenges, we propose a boundary-aware cross-scale context refinement network (BSCNet) based on SegFormer-B0. First, a shallow multiscale boundary auxiliary supervision (SMBAS) module uses high-resolution features from the first two encoder stages to impose boundary constraints during training. Second, a cross-scale decoder feature alignment (CSDA) module models semantic consistency and injects feature differences through residual connections to progressively align features from deep to shallow levels before fusion. Third, a multisource context adaptive refinement (MCAR) module integrates image-level semantics with regional context from multiple receptive fields to refine the fused features. BSCNet achieves a mean intersection over union (mIoU) of 67.95% and a mean class accuracy (mAcc) of 77.17% on the Semantic Segmentation of Underwater Imagery (SUIM) validation set, together with an mIoU of 69.39% on the DUT-USEG validation set. Ablation experiments support the effectiveness of the individual modules and their complete joint configuration under the experimental settings used in this study.

## Modules

### SMBAS — Shallow Multiscale Boundary Auxiliary Supervision

SMBAS jointly uses high-resolution features from the first two encoder stages and explicit boundary labels to improve the shared encoder's perception of weak underwater contours during training. The auxiliary boundary branch is not used during inference and therefore adds no inference overhead.

### CSDA — Cross-Scale Decoder Feature Alignment

CSDA combines semantic consistency mapping with residual injection of feature differences to progressively align features from deep to shallow levels before concatenation. This reduces the discrepancy between coarse high-level semantics and intermediate spatial details.

### MCAR — Multisource Context Adaptive Refinement

MCAR supplements fused features with image-level scene semantics and regional context from multiple receptive fields. A zero-initialized learnable residual coefficient controls the strength of context injection to improve semantic discrimination in complex underwater regions.

## Dataset Sources

- **SUIM (Semantic Segmentation of Underwater Imagery):** [Official dataset page](https://irvlab.cs.umn.edu/resources/suim-dataset)
- **DUT-USEG:** [Official dataset repository](https://github.com/baxiyi/DUT-USEG)
