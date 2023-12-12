default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend', _scope_='mmseg'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer',
    _scope_='mmseg')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel', _scope_='mmseg')
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, _scope_='mmseg')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None,
    _scope_='mmseg')
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False,
        _scope_='mmseg'),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
val_cfg = dict(type='ValLoop', _scope_='mmseg')
test_cfg = dict(type='TestLoop', _scope_='mmseg')
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmseg'),
    logger=dict(type='LoggerHook', interval=25, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmseg'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,
        by_epoch=False,
        max_keep_ckpts=5,
        save_last=True,
        save_best='mIoU',
        rule='greater',
        published_keys=[
            'meta',
            'state_dict',
        ]),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmseg'),
    visualization=dict(type='SegVisualizationHook', _scope_='mmseg'))
img_dir = 'yolo_sn'
ann_dir = 'masks'
crop_size = (
    512,
    512,
)
scale = (
    1024,
    1024,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(
            1024,
            1024,
        ),
        ratio_range=(
            0.5,
            2.0,
        ),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(
        512,
        512,
    ), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(
        1024,
        1024,
    ), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        data_prefix=dict(
            img_path='yolo_sn/train', seg_map_path='masks/train/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomResize',
                scale=(
                    1024,
                    1024,
                ),
                ratio_range=(
                    0.5,
                    2.0,
                ),
                keep_ratio=True),
            dict(
                type='RandomCrop', crop_size=(
                    512,
                    512,
                ), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackSegInputs'),
        ],
        type='MoNuSegDataset',
        data_root='/mnt/data/joren/datasets/MoNuSeg'),
    batch_size=2)
val_dataloader = dict(
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_prefix=dict(img_path='yolo_sn/val', seg_map_path='masks/val/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                1024,
                1024,
            ), keep_ratio=False),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='MoNuSegDataset',
        data_root='/mnt/data/joren/datasets/MoNuSeg'))
test_dataloader = dict(
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_prefix=dict(img_path='yolo_sn/test', seg_map_path='masks/test/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                1024,
                1024,
            ), keep_ratio=False),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='MoNuSegDataset',
        data_root='/mnt/data/joren/datasets/MoNuSeg'))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mIoU',
    ])
test_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mIoU',
    ])
norm_cfg = dict(type='SyncBN', requires_grad=True, _scope_='mmseg')
data_preprocessor = dict(
    size=(
        512,
        512,
    ),
    type='SegDataPreProcessor',
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    _scope_='mmseg')
model = dict(
    data_preprocessor=dict(
        size=(
            512,
            512,
        ),
        type='SegDataPreProcessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    test_cfg=dict(mode='whole'),
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        dilations=(
            1,
            1,
            2,
            4,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_classes=19,
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_classes=19,
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    train_cfg=dict(),
    _scope_='mmseg')
dataset_type = 'MoNuSegDataset'
data_root = '/mnt/data/joren/datasets/MoNuSeg'
work_dir = './work_dirs/MoNuSeg/deeplabv3plus/2023-07-31_22_12_06/'
randomness = dict(seed=0)
