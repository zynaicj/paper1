_base_ = [
    './segformer_mit-b0_useg_baseline_ft40k.py',
]

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1500),
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

model = dict(
    auxiliary_head=dict(
        _delete_=True,
        type='BoundaryHead',
        in_channels=[32, 64],
        in_index=[0, 1],
        channels=64,
        num_classes=1,
        # USEG boundary pixels are only about 0.2% of the mask. BoundaryLoss
        # balances positive and negative pixels per batch; the multiplier
        # restores a useful gradient scale after that normalization.
        loss_decode=dict(type='BoundaryLoss', loss_weight=20.0)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadBoundaryMask'),
    dict(
        type='RandomResize',
        scale=(853, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(480, 640), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

work_dir = (
    './work_dirs/useg/'
    'segformer_b0_useg_boundary_balanced_from_last_160k_seed0')

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir=(
            'experiments/useg/'
            'segformer_b0_useg_boundary_balanced_from_last_160k_seed0'),
        config=(
            'configs/segformer/'
            'segformer_mit-b0_useg_boundary_balanced_from_last_160k.py'),
        name='USEG Balanced Boundary from Final Baseline 160k Seed 0',
        note=(
            'Initialized from the final 160k baseline checkpoint; uses '
            'per-batch balanced boundary supervision and low-LR fine-tuning.'))
]
