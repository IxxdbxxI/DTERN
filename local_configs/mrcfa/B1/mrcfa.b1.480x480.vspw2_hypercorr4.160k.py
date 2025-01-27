_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/vspw_repeat2.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings 测试设计的多尺度融合模块以及PID的融合模块，CAT模块，块内块间计算
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder_clips',
    # pretrained='/cluster/work/cvl/celiuce/video-seg/models/segformer/pretrained_models/mit_b1.pth',
    pretrained='/root/workspace/XU/Code/VSS-MRCFA-main/pretrained_model/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch'),
    decode_head=dict(
        # SegFormerHead_clips2_resize_1_8_hypercorrelation2_topk_ensemble4
        # type='SegFormerHead_clips2_resize_1_8_hypercorrelation2_topk_ensemble4',  # SegFormerHead_clips2_resize_1_8_hypercorrelation2_ensemble5
        # type='SegFormerHead_clips2_resize_1_8_hypercorrelation2_topk_ensemble4_FeaturePyramid_baseline',
        # type='SegFormerHead_clips2_resize_1_8_hypercorrelation2_topk_ensemble4_FeaturePyramid_baseline2',
        type='SegFormerHead_clips2_resize_1_8_static_dynamic_ensemble4_3',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=124,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256,lv=[[1,1],[2,2],[4,4],[8,8]]),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # loss_decode=[dict(type='CrossEntropyLoss', loss_name='ce_loss',use_sigmoid=False,loss_weight=1.0),dict(type='DiceLoss', loss_name='dice_loss',loss_weight=3.,eps=1.)],
        num_clips=4,
        hypercorre=True,
        backbone='b1'),
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
