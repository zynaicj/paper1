_base_ = ['./segformer_mit-b0_suim_boundary_residual_aspp.py']

model = dict(
    decode_head=dict(
        type='SegformerResidualGlobalOnlyHead',
        aspp_channels=128,
        aspp_init_gamma=0.0,
    ),
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
            'decode_head.residual_aspp': dict(lr_mult=10.),
        }))

load_from = (
    'experiments/module_a/segformer_b0_suim_boundary_w10/checkpoints/'
    'best_mIoU_iter_80000.pth')
resume = False
randomness = dict(seed=0, deterministic=False)

work_dir = './work_dirs/module_b/segformer_b0_suim_boundary_global_only'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir='experiments/module_b/segformer_b0_suim_boundary_global_only',
        config='configs/segformer/segformer_mit-b0_suim_boundary_global_only.py',
        name='Module A + MCAR Global-Only Ablation',
        note='MCAR context-source ablation: residual image-level global pooling branch only.',
    )
]
