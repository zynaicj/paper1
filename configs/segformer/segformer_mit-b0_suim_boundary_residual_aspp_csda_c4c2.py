_base_ = ['./segformer_mit-b0_suim_boundary_residual_aspp_sgpcf.py']

model = dict(
    decode_head=dict(
        sg_fuse_stages=('c4_c2', ),
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
            'decode_head.sg_fuses': dict(lr_mult=10.),
        }))

load_from = (
    'experiments/module_b/segformer_b0_suim_boundary_residual_aspp/'
    'best_mIoU_iter_130000.pth')
resume = False
randomness = dict(seed=0, deterministic=False)

work_dir = './work_dirs/module_c/segformer_b0_suim_boundary_residual_aspp_csda_c4c2'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir='experiments/module_c/segformer_b0_suim_boundary_residual_aspp_csda_c4c2',
        config='configs/segformer/segformer_mit-b0_suim_boundary_residual_aspp_csda_c4c2.py',
        name='Module A+B + CSDA C4-to-C2 Ablation',
        note='CSDA progressive-transfer ablation: cross-level c4_c2 direct semantic-guided fusion.',
    )
]
