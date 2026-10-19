_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/suim.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (480, 640)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32
)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mit_b0_20220624-7e0fe6dd.pth'
        )
    ),
    decode_head=dict(num_classes=8),
    # ─── Boundary Head ───
    auxiliary_head=dict(
        type='BoundaryHead',
        in_channels=[32, 64],
        in_index=[0, 1],
        channels=64,
        num_classes=1,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0
        )
    ),
    test_cfg=dict(type='whole')
)

# ─── Train pipeline：加 LoadBoundaryMask ───
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadBoundaryMask'),
    dict(type='RandomResize', scale=(640, 480), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# ─── Val pipeline ───
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 480), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(pipeline=val_pipeline)
)
test_dataloader = val_dataloader

# ─── 加载 baseline 权重 ───
load_from = 'experiments/module_a/segformer_b0_suim8_baseline_finall/checkpoints/best_mIoU_iter_160000.pth'

# ─── 优化器 ───
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
        })
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        save_best='mIoU',
        rule='greater',
        save_last=True,
        max_keep_ckpts=3
    )
)
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=10000
)

work_dir = './work_dirs/module_a/segformer_b0_suim_boundary'
