_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/vspw_repeat2.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True


num_cluster = 250
model = dict(
    type='EncoderDecoder_clips',
    pretrained='/root/workspace/XU/Code/VSS-MRCFA-main/pretrained_model/mit_b0.pth',
    backbone=dict(
        type='mit_b0',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead_clips2_resize_1_8_Cluster_SegDeformer_ensemble4',
        in_channels=[32, 64, 160, 256],
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
        backbone='b0',
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


data = dict(samples_per_gpu=2)
evaluation = dict(interval=160000, metric='mIoU')
