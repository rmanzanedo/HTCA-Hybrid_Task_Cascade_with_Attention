_base_ = './htc_effi_without_semantic.py'

# model = dict(
#     type='HybridTaskCascade',
#     backbone=dict(
#         type='EfficientDet',
#         # depth=50,
#         num_stages=4,
#         # out_indices=(0, 1, 2, 3),
#         # frozen_stages=1,
#         # norm_cfg=dict(type='BN', requires_grad=True),
#         # norm_eval=True,
#         style='pytorch',
#         # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
#     neck=dict(
#         type='FPN',
#         in_channels=[40, 384, 384, 384],
#         out_channels=256,
#         num_outs=5)))

# learning policy
# lr_config = dict(step=[19, 23, 25, 28])
# runner = dict(type='EpochBasedRunner', max_epochs=30)
runner = dict(type='EpochBasedRunner', max_epochs=12)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

# _base_ = './htc_effi_without_semantic.py'

model = dict(
    backbone=dict(
        # out_features=['p2'],),
        out_features=['p2', 'p3', 'p4', 'p5'],),
    # rpn_head=None,
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            # strides=[4]),),
            strides=[4, 8, 16, 32]),),
    roi_head=dict(
    mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            # featmap_strides=[4]),),
            featmap_strides=[4, 8, 16, 32]),
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            # featmap_strides=[4]),
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=183,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2))),
    neck = dict(
        type='ChannelMapper',
    #     # in_channels=[40, 384, 384, 384],
        in_channels=[32, 160, 160, 160],
        out_channels=256,
        num_outs=4))
    # neck = None)
data_root = '/disk2/datasets/coco/'
img_norm_cfg = dict(
    mean=[0,0,0], std=[1,1,1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        seg_prefix=data_root + 'stuffthingmaps/train2017/',
        pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

load_from ='../checkpoints/htc_for_efficientNet.pth'