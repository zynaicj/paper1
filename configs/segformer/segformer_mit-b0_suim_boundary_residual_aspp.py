_base_ = [
    './segformer_mit-b0_suim_boundary.py',
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
    auxiliary_head=dict(
        _delete_=True,
        type='BoundaryHead',
        in_channels=[32, 64],
        in_index=[0, 1],
        channels=64,
        num_classes=1,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
)

# Continue from the best Module A checkpoint, then train the ASPP context block.
load_from = 'experiments/module_a/segformer_b0_suim_boundary_w10/checkpoints/best_mIoU_iter_80000.pth'

work_dir = './work_dirs/module_b/segformer_b0_suim_boundary_residual_aspp'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir='experiments/module_b/segformer_b0_suim_boundary_residual_aspp',
        config='configs/segformer/segformer_mit-b0_suim_boundary_residual_aspp.py',
        name='Module A + Residual ASPP',
        note='BoundaryHead w10 with residual ASPP after SegFormer fusion.',
    )
]
