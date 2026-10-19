_base_ = [
    './segformer_mit-b0_suim_boundary_residual_aspp.py',
]

model = dict(
    decode_head=dict(
        type='SegformerResidualASPPSGPCFHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=8,
        dropout_ratio=0.1,
        align_corners=False,
        aspp_channels=128,
        dilations=(1, 6, 12, 18),
        aspp_init_gamma=0.0,
        sg_channels=256,
        sg_mid_channels=128,
        sg_init_gamma=0.0,
        sg_fuse_stages=('c4_c3', 'c3_c2'),
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

# Match the original A+B batch-4 training conditions for a full 160k
# SG-PCF verification run.
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        save_best='mIoU',
        rule='greater',
        save_last=True,
        max_keep_ckpts=3))

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=10000)

# Continue from the best Module A+B checkpoint. SG-PCF starts as identity
# because sg_init_gamma=0.0, so only its new parameters should be missing.
load_from = 'experiments/module_b/segformer_b0_suim_boundary_residual_aspp/best_mIoU_iter_130000.pth'

work_dir = './work_dirs/module_c/segformer_b0_suim_boundary_residual_aspp_sgpcf_bs4_160k'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir='experiments/module_c/segformer_b0_suim_boundary_residual_aspp_sgpcf_bs4_160k',
        config='configs/segformer/segformer_mit-b0_suim_boundary_residual_aspp_sgpcf.py',
        name='Module A + Residual ASPP + SG-PCF BS4 160k',
        note='C-v2 SG-PCF with original A+B batch-4 training style; 160k iterations from A+B best checkpoint.',
    )
]
