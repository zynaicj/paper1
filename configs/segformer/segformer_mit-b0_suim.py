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
    # 既然图片已经是 640x480，这里设为 480, 640 是匹配的
    # size=crop_size,
    size_divisor=32
)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    # backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mit_b0_20220624-7e0fe6dd.pth'
        )
    ),
    decode_head=dict(num_classes=8),
    test_cfg=dict(type='whole')

    # auxiliary_head=dict(
    #     type='BoundaryHead',

    #     in_channels=[32, 64],
    #     in_index=[0, 1],

    #     channels=64,

    #     num_classes=1,

    #     loss_decode=dict(
    #         type='CrossEntropyLoss',
    #         use_sigmoid=True,
    #         loss_weight=2
    #     )
    # )
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
# train_dataloader = dict(batch_size=8, num_workers=4)
# val_dataloader = dict(
#     batch_size=1, num_workers=4
#     )
# test_dataloader = val_dataloader

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

# train_pipeline = [

#     dict(type='LoadImageFromFile'),

#     dict(type='LoadAnnotations'),

#     dict(type='LoadBoundaryMask'),

#     dict(
#         type='RandomResize',
#         scale=(2048, 512),
#         ratio_range=(0.5, 2.0),
#         keep_ratio=True
#     ),

#     dict(
#         type='RandomCrop',
#         crop_size=crop_size
#     ),

#     dict(type='RandomFlip', prob=0.5),

#     dict(type='PhotoMetricDistortion'),

#     dict(type='PackBoundaryInputs')
# ]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # 建议先缩放到略大，再裁剪，效果比直接 Resize 更好
    dict(type='RandomResize', scale=(640, 480), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,

    dataset=dict(
        pipeline=train_pipeline
    )
)

# =========================
# val pipeline
# =========================

val_pipeline = [
    dict(type='LoadImageFromFile'),
    # 依然保留 Resize 作为“保险”，但 scale 必须严格对应你离线处理后的尺寸
    # 注意：mmseg 中 scale 习惯写成 (width, height)
    dict(type='Resize', scale=(640, 480), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = val_pipeline

test_pipeline = val_pipeline

# =========================
# val dataloader
# =========================

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        pipeline=val_pipeline
    )
)

test_dataloader = val_dataloader



work_dir = './work_dirs/segformer_b0_suim8_baseline_finall'
