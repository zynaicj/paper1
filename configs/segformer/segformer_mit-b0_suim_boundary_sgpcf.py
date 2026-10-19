_base_ = [
    './segformer_mit-b0_suim_boundary.py',
]

# Strict A+C ablation:
#   A: BoundaryHead supervision on c1+c2
#   C: SG-PCF progressive semantic fusion (c4->c3, c3->c2)
# No Residual ASPP.
model = dict(
    decode_head=dict(
        _delete_=True,
        type='SegformerSGPCFHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=8,
        dropout_ratio=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0)))

# Continue from the official Module A checkpoint.
load_from = (
    'experiments/module_a/segformer_b0_suim_boundary_w10/'
    'checkpoints/best_mIoU_iter_80000.pth')

work_dir = './work_dirs/module_c/segformer_b0_suim_boundary_sgpcf_bs4_160k'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir=(
            'experiments/module_c/'
            'segformer_b0_suim_boundary_sgpcf_bs4_160k'),
        config='configs/segformer/segformer_mit-b0_suim_boundary_sgpcf.py',
        name='SUIM Module A + SG-PCF BS4 160k',
        note=(
            'Strict A+C ablation: BoundaryHead w10 + SG-PCF C-v2, '
            'without Residual ASPP.'))
]
