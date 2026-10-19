# Copyright (c) OpenMMLab. All rights reserved.
from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .ddr_head import DDRHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .dpt_head import DPTHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .encoder_feature_aux_head import EncoderFeatureAuxHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .ham_head import LightHamHead
from .isa_head import ISAHead
from .knet_head import IterativeDecodeHead, KernelUpdateHead, KernelUpdator
from .lraspp_head import LRASPPHead
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .pid_head import PIDHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .san_head import SideAdapterCLIPHead
from .segformer_head import SegformerHead
from .segformer_cacr_dgpf_head import (
    SegformerCACRDGPFHead, SegformerCACRHead, SegformerDGPFHead,
    SegformerResidualASPPDGPFHead)
from .segformer_freq_fusion_head import SegformerFreqFusionHead
from .segformer_gated_freq_fusion_head import (
    SegformerGatedFreqFusionHead, SegformerSemanticGuidedFreqFusionHead)
from .segformer_residual_aspp_head import (
    SegformerResidualASPPHead, SegformerResidualGlobalOnlyHead,
    SegformerResidualRegionalOnlyHead)
from .segformer_residual_aspp_fdf_head import (
    SegformerResidualASPPFDFHead)
from .segformer_residual_aspp_sgpcf_head import (
    SegformerResidualASPPBPSFHead, SegformerResidualASPPSGPCFHead,
    SegformerSGPCFHead)
from .segformer_strip_pooling_head import SegformerStripPoolingHead
from .upcm import UPCM
from .segmenter_mask_head import SegmenterMaskTransformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .stdc_head import STDCHead
from .uper_head import UPerHead
from .vpd_depth_head import VPDDepthHead
from .boundary_head import BoundaryHead
from .boundary_guided_head import BoundaryGuidedSegformerHead
from .global_head import GlobalSemanticHead
from .hard_class_head import HardClassSemanticHead
from .ff_boundary_head import FFBoundaryHead
from .freq_fusion_decoder import FreqFusionDecoder

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
    'SETRMLAHead', 'DPTHead', 'SETRMLAHead', 'SegmenterMaskTransformerHead',
    'SegformerHead', 'SegformerFreqFusionHead', 'SegformerGatedFreqFusionHead',
    'SegformerSemanticGuidedFreqFusionHead', 'SegformerResidualASPPHead',
    'SegformerResidualGlobalOnlyHead', 'SegformerResidualRegionalOnlyHead',
    'SegformerCACRHead', 'SegformerDGPFHead', 'SegformerCACRDGPFHead',
    'SegformerResidualASPPDGPFHead', 'SegformerResidualASPPSGPCFHead',
    'SegformerResidualASPPBPSFHead', 'SegformerSGPCFHead',
    'SegformerStripPoolingHead', 'ISAHead',
    'STDCHead', 'IterativeDecodeHead', 'KernelUpdateHead', 'KernelUpdator',
    'MaskFormerHead', 'Mask2FormerHead', 'LightHamHead', 'PIDHead', 'DDRHead',
    'VPDDepthHead',
    'SideAdapterCLIPHead', 'BoundaryHead', 'BoundaryGuidedSegformerHead',
    'FFBoundaryHead', 'FreqFusionDecoder', 'GlobalSemanticHead',
    'HardClassSemanticHead', 'UPCM', 'EncoderFeatureAuxHead',
]
