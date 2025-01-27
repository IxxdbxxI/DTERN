_base_ = [
    '../../_base_/models/segformer.py',
    # '../../_base_/datasets/ade20k_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# data settings
dataset_type = 'VSPWDataset2'
data_root = 'data/vspw/VSPW_480p'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (480, 480)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(type='Resize', img_scale=(2048, 640), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(853, 480), ratio_range=(0.5, 2.0), process_clips=True),
    dict(type='RandomCrop_clips', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_clips', prob=0.5),
    dict(type='PhotoMetricDistortion_clips'),
    dict(type='Normalize_clips', **img_norm_cfg),
    dict(type='Pad_clips', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle_clips'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 640),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='AlignedResize', keep_ratio=True, size_divisor=32), # Ensure the long and short sides are divisible by 32
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(853, 480),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='AlignedResize_clips', keep_ratio=True, size_divisor=32), # Ensure the long and short sides are divisible by 32
            dict(type='RandomFlip_clips'),
            dict(type='Normalize_clips', **img_norm_cfg),
            dict(type='ImageToTensor_clips', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    # workers_per_gpu=4,
    # train=dict(
    #     type='RepeatDataset',
    #     times=50,
    #     dataset=dict(
    #         type=dataset_type,
    #         data_root=data_root,
    #         img_dir='images/training',
    #         ann_dir='annotations/training',
    #         pipeline=train_pipeline)),
    # val=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     img_dir='images/validation',
    #     ann_dir='annotations/validation',
    #     pipeline=test_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     img_dir='images/validation',
    #     ann_dir='annotations/validation',
    #     pipeline=test_pipeline))
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotations/training',
            split='train',
            pipeline=train_pipeline,
            dilation=[-9,-6,-3])),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        split='val',
        pipeline=test_pipeline,
        dilation=[-9,-6,-3]),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        split='val',
        pipeline=test_pipeline,
        dilation=[-9,-6,-3]))

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True


num_cluster = 124
model = dict(
    type='EncoderDecoder_clips',
    pretrained='/root/workspace/XU/Code/VSS-MRCFA-main/pretrained_model/mit_b5.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead_clips2_resize_1_8_Cluster_SegDeformer_ensemble4',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=124,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        num_clips=4,
        hypercorre=True,
        cityscape=False,
        backbone='b5',
        cross_method = 'Cluster', #['CAT', 'Focal','Focal_CAT','cluster'] cluster is for loss cal
        num_cluster = num_cluster,
        # loss_cluster = dict(type='ClusterLoss',class_num = num_cluster, temperature = 0.07),
        # loss_cluster = dict(type='NTXentLoss',bs=num_cluster,tau=0.5, cos_sim=True),
        loss_cluster = dict(type='DCL',temperature=0.1),
        # loss_cluster = dict(type='DCLW',sigma=0.5,temperature=0.1),
        need_cluster_loss = False, 
        cluster_with_t = False, #default is False
        need_segdeformer = True, #default is true, 测试聚类是否需要特定的解码器结构
        aux_loss_decode = False
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

evaluation = dict(interval=210000, metric='mIoU')
