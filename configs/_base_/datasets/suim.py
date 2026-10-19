# dataset settings

dataset_type = 'SUIMDataset'

data_root = 'data/SUIM'

crop_size = (480, 640)

# =========================
# Train pipeline
# =========================

train_pipeline = [

    dict(type='LoadImageFromFile'),

    dict(type='LoadAnnotations'),

    dict(
        type='RandomResize',
        scale=(640, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True
    ),

    dict(
        type='RandomCrop',
        crop_size=crop_size
    ),

    dict(type='RandomFlip', prob=0.5),

    dict(type='PhotoMetricDistortion'),

    dict(type='PackSegInputs')
]

# =========================
# Test pipeline
# =========================

test_pipeline = [

    dict(type='LoadImageFromFile'),

    dict(
        type='Resize',
        scale=(640, 480),
        keep_ratio=True
    ),

    dict(type='LoadAnnotations'),

    dict(type='PackSegInputs')
]

# =========================
# Train dataloader
# =========================

train_dataloader = dict(

    batch_size=4,

    num_workers=4,

    persistent_workers=True,

    sampler=dict(
        type='InfiniteSampler',
        shuffle=True
    ),

    dataset=dict(

        type=dataset_type,

        data_root=data_root,

        ann_file='splits/train.txt',

        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='Segmentation'
        ),

        pipeline=train_pipeline
    )
)

# =========================
# Validation dataloader
# =========================

val_dataloader = dict(

    batch_size=1,

    num_workers=2,

    persistent_workers=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),

    dataset=dict(

        type=dataset_type,

        data_root=data_root,

        ann_file='splits/val.txt',

        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='Segmentation'
        ),

        pipeline=test_pipeline
    )
)

# =========================
# Test dataloader
# =========================

test_dataloader = val_dataloader

# =========================
# Evaluator
# =========================

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU']
)

test_evaluator = val_evaluator
