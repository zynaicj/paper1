_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/useg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

crop_size = (480, 640)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mit_b0_20220624-7e0fe6dd.pth')),
    decode_head=dict(num_classes=5),
    auxiliary_head=None,
    test_cfg=dict(type='whole'))

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

work_dir = './work_dirs/useg/segformer_b0_useg_baseline'

custom_hooks = [
    dict(
        type='ArchiveExperimentHook',
        out_dir='experiments/useg/segformer_b0_useg_baseline',
        config='configs/segformer/segformer_mit-b0_useg.py',
        name='USEG Baseline SegFormer-B0',
        note='MIT-B0 pretrained backbone with SegFormerHead on USEG 5-class masks.')
]
