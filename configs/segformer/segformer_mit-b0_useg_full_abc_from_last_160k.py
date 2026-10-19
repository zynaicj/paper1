_base_ = [
    './segformer_mit-b0_useg_boundary_balanced_from_last_160k.py',
]

# Full model:
#   A: balanced BoundaryHead supervision on c1+c2
#   B: residual ASPP after SegFormer multi-scale fusion
#   C: SG-PCF C-v2 progressive semantic fusion (c4->c3, c3->c2)
model = dict(
    decode_head=dict(
        _delete_=True,
        type='SegformerResidualASPPSGPCFHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=5,
        dropout_ratio=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        aspp_channels=128,
        dilations=(1, 6, 12, 18),
        aspp_init_gamma=0.0,
        sg_channels=256,
        sg_mid_channels=128,
        sg_init_gamma=0.0,
        sg_fuse_stages=('c4_c3', 'c3_c2'),
        sg_sim_type='spatial'),
    auxiliary_head=dict(
        _delete_=True,
        type='BoundaryHead',
        in_channels=[32, 64],
        in_index=[0, 1],
        channels=64,
        num_classes=1,
        loss_decode=dict(type='BoundaryLoss', loss_weight=20.0)))

# Preserve the final baseline backbone and decoder with a low LR while giving
# the newly initialized A/B/C parameters enough room to learn.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.000006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'decode_head': dict(lr_mult=1.),
            'decode_head.residual_aspp': dict(lr_mult=10.),
            'decode_head.sg_fuses': dict(lr_mult=10.),
            'auxiliary_head': dict(lr_mult=10.),
        }))

work_dir = './work_dirs/useg/segformer_b0_useg_full_abc_from_last_160k_seed0'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir=(
            'experiments/useg/'
            'segformer_b0_useg_full_abc_from_last_160k_seed0'),
        config=(
            'configs/segformer/'
            'segformer_mit-b0_useg_full_abc_from_last_160k.py'),
        name='USEG Full A+B+C from Final Baseline 160k Seed 0',
        note=(
            'Full model initialized from the final 160k USEG baseline: '
            'balanced BoundaryHead + Residual ASPP + SG-PCF C-v2.'))
]
