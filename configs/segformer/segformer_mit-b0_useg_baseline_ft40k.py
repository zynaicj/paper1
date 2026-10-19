_base_ = [
    './segformer_mit-b0_useg.py',
]

# Short, low-LR control run initialized from the final saved USEG baseline.
# Use load_from rather than --resume so this starts a fresh 40k schedule.
load_from = 'experiments/useg/segformer_b0_useg_baseline/iter_160000.pth'

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
            'auxiliary_head': dict(lr_mult=10.),
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=40000,
        by_epoch=False)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        save_best='mIoU',
        rule='greater',
        save_last=True,
        max_keep_ckpts=2))

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,
    val_interval=5000)

randomness = dict(seed=0, deterministic=False)

work_dir = './work_dirs/useg/segformer_b0_useg_baseline_last_ft40k_seed0'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir=(
            'experiments/useg/'
            'segformer_b0_useg_baseline_last_ft40k_seed0'),
        config='configs/segformer/segformer_mit-b0_useg_baseline_ft40k.py',
        name='USEG Final Baseline Fine-tune 40k Seed 0',
        note=(
            'Low-LR control run initialized from the final 160k USEG '
            'baseline checkpoint.'))
]
