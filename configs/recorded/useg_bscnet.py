# Recorded training configuration; only the machine-specific project prefix was removed.
crop_size = (
    480,
    640,
)
custom_hooks = [
    dict(
        config=
        'configs/segformer/segformer_mit-b0_useg_full_abc_from_last_160k.py',
        name='USEG Full A+B+C from Final Baseline 160k Seed 0',
        note=
        'Full model initialized from the final 160k USEG baseline: balanced BoundaryHead + Residual ASPP + SG-PCF C-v2.',
        out_dir=
        'experiments/useg/segformer_b0_useg_full_abc_from_last_160k_seed0',
        type='ArchiveExperimentHook'),
]
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/USEG'
dataset_type = 'USEGDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=10000,
        max_keep_ckpts=3,
        rule='greater',
        save_best='mIoU',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    853,
    480,
)
launcher = 'none'
load_from = 'experiments/useg/segformer_b0_useg_baseline/iter_160000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        channels=64,
        in_channels=[
            32,
            64,
        ],
        in_index=[
            0,
            1,
        ],
        loss_decode=dict(loss_weight=20.0, type='BoundaryLoss'),
        num_classes=1,
        type='BoundaryHead'),
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=32,
        in_channels=3,
        init_cfg=dict(
            checkpoint='checkpoints/mit_b0_20220624-7e0fe6dd.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        aspp_channels=128,
        aspp_init_gamma=0.0,
        channels=256,
        dilations=(
            1,
            6,
            12,
            18,
        ),
        dropout_ratio=0.1,
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=5,
        sg_channels=256,
        sg_fuse_stages=(
            'c4_c3',
            'c3_c2',
        ),
        sg_init_gamma=0.0,
        sg_mid_channels=128,
        sg_sim_type='spatial',
        type='SegformerResidualASPPSGPCFHead'),
    pretrained=None,
    test_cfg=dict(mode='whole', type='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-06, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict({
            'auxiliary_head': dict(lr_mult=10.0),
            'decode_head': dict(lr_mult=1.0),
            'decode_head.residual_aspp': dict(lr_mult=10.0),
            'decode_head.sg_fuses': dict(lr_mult=10.0),
            'norm': dict(decay_mult=0.0),
            'pos_block': dict(decay_mult=0.0)
        })),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1500, start_factor=0.1, type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=160000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
randomness = dict(deterministic=False, seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/Segmentation/val.txt',
        data_prefix=dict(img_path='JPEGImages', seg_map_path='Segmentation'),
        data_root='data/USEG',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                853,
                480,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='USEGDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        853,
        480,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=160000, type='IterBasedTrainLoop', val_interval=10000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='ImageSets/Segmentation/train.txt',
        data_prefix=dict(img_path='JPEGImages', seg_map_path='Segmentation'),
        data_root='data/USEG',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='LoadBoundaryMask'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    853,
                    480,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    480,
                    640,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='USEGDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadBoundaryMask'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            853,
            480,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        480,
        640,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/Segmentation/val.txt',
        data_prefix=dict(img_path='JPEGImages', seg_map_path='Segmentation'),
        data_root='data/USEG',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                853,
                480,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='USEGDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/useg/segformer_b0_useg_full_abc_from_last_160k_seed0'
