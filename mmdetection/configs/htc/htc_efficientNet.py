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
lr_config = dict(step=[16, 19, 24, 28])
runner = dict(type='EpochBasedRunner', max_epochs=30)

# _base_ = './htc_effi_without_semantic.py'
model = dict(
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
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
        type='FPN',
        # in_channels=[40, 384, 384, 384],
        in_channels=[32, 160, 160, 160],
        out_channels=256,
        num_outs=5))
data_root = '/disk2/datasets/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    train=dict(
        seg_prefix=data_root + 'stuffthingmaps/train2017/',
        pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))