_base_ = [
    './segformer_mit-b0_suim.py',
]

# Strict C-only ablation:
#   C: SG-PCF progressive semantic fusion (c4->c3, c3->c2)
# No BoundaryHead and no Residual ASPP.
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
    auxiliary_head=None)

# Continue from the official SUIM baseline checkpoint.
load_from = (
    'experiments/module_a/segformer_b0_suim8_baseline_finall/'
    'checkpoints/best_mIoU_iter_160000.pth')

work_dir = './work_dirs/module_c/segformer_b0_suim_sgpcf_bs4_160k'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir='experiments/module_c/segformer_b0_suim_sgpcf_bs4_160k',
        config='configs/segformer/segformer_mit-b0_suim_sgpcf.py',
        name='SUIM Baseline + SG-PCF BS4 160k',
        note=(
            'Strict C-only ablation: SG-PCF C-v2 progressive semantic '
            'fusion, without BoundaryHead or Residual ASPP.'))
]
