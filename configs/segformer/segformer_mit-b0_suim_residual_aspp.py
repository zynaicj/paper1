_base_ = [
    './segformer_mit-b0_suim.py',
]

model = dict(
    decode_head=dict(
        type='SegformerResidualASPPHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=8,
        dropout_ratio=0.1,
        align_corners=False,
        aspp_channels=128,
        dilations=(1, 6, 12, 18),
        aspp_init_gamma=0.0,
    ),
    auxiliary_head=None,
)

# Module B alone: continue from the best SegFormer-B0 baseline checkpoint.
load_from = 'experiments/module_a/segformer_b0_suim8_baseline_finall/checkpoints/best_mIoU_iter_160000.pth'

work_dir = './work_dirs/module_b/segformer_b0_suim_residual_aspp'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir='experiments/module_b/segformer_b0_suim_residual_aspp',
        config='configs/segformer/segformer_mit-b0_suim_residual_aspp.py',
        name='Baseline + Residual ASPP',
        note='SegFormer-B0 baseline with residual ASPP after SegFormer fusion; no BoundaryHead.',
    )
]
