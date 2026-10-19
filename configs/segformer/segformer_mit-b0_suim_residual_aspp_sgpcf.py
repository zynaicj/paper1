_base_ = [
    './segformer_mit-b0_suim_residual_aspp.py',
]

# Strict B+C ablation:
#   B: Residual ASPP after SegFormer fusion
#   C: SG-PCF progressive semantic fusion (c4->c3, c3->c2)
# No BoundaryHead.
model = dict(
    decode_head=dict(
        _delete_=True,
        type='SegformerResidualASPPSGPCFHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=8,
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
    auxiliary_head=None)

# Continue from the official Module B checkpoint.
load_from = (
    'experiments/module_b/segformer_b0_suim_residual_aspp/'
    'best_mIoU_iter_70000.pth')

work_dir = './work_dirs/module_c/segformer_b0_suim_residual_aspp_sgpcf_bs4_160k'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir=(
            'experiments/module_c/'
            'segformer_b0_suim_residual_aspp_sgpcf_bs4_160k'),
        config=(
            'configs/segformer/'
            'segformer_mit-b0_suim_residual_aspp_sgpcf.py'),
        name='SUIM Module B + SG-PCF BS4 160k',
        note=(
            'Strict B+C ablation: Residual ASPP + SG-PCF C-v2, '
            'without BoundaryHead or boundary-mask loading.'))
]
