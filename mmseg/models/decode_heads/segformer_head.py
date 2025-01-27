# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import collections
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead, BaseDecodeHead_clips,BaseDecodeHead_clips2
from mmseg.models.utils import *
import attr



from IPython import embed

import cv2
# from .hypercorre import hypercorre_topk2
# from .hypercorre2 import hypercorre_topk2 as hypercorre_topk21 # linear_qkv_sim_thw
# from .hypercorre3 import hypercorre_topk2 as hypercorre_topk31 #cnn_qk_liner_v_sim_thw
# from .hypercorre4 import hypercorre_topk2 as hypercorre_topk41 #cnn_qk_cnn_v_sim_hw_out_t
# from .hypercorre5 import hypercorre_topk2 as hypercorre_topk51 #CAT blocks for cross and time refiner(dvis)
# from .hypercorre6 import hypercorre_topk2 as hypercorre_topk61 #CAT blocks and Segdeformer for time and ratio fusion all
# from .fusion_hypercorre5_6 import hypercorre_topk2 as hypercorre_topk71 #CAT_blk and PAGFM for ratio fusion and Segdeformer  for time fusion
# from .hypercorre7_maskguide import hypercorre_topk2 as hypercorre_topk81 #CAT blocks and Segdeformer for time and ratio fusion all and mask guided
from .hypercorre8_cluster_t import hypercorre_topk2 as hypercorre_topk91 #Cluster Block(paca_vit) for t and Segdeformer 
# from .hypercorre8_cluster_t_CL import hypercorre_topk2 as hypercorre_topk92 #Cluster using CL 

# from .hypercorre8_cluster2_t import hypercorre_topk2 as hypercorre_topk101 #Cluster Block(cluster former) for t and Segdeformer 
# # from .hypercorre8_cluster3_t import hypercorre_topk2 as hypercorre_topk101 #Cluster Block(cluster former) for t and Segdeformer (layer2 加了mem交互,transseg将聚类数目修改由150为124)

# from .hypercorre8_cluster4_t import hypercorre_topk2 as hypercorre_topk111 #Cluster Block(cluster former) for t and Segdeformer，加入近亲融合，将flash_atten修改成center之间的,这个拉近了类间距

# from .hypercorre8_cluster5_t import hypercorre_topk2 as hypercorre_topk121 #融合paca-vit的聚类和cluster-former的聚类分配，没有融合t,聚类数目是t*n_cluster
# from .hypercorre8_cluster6_t import hypercorre_topk2 as hypercorre_topk131 #先融合时间t（对t加权求和），再更新cluster，cluster-former的聚类分配,聚类数目是n_cluster
# from .hypercorre10 import hypercorre_topk2 as hypercorre_topk10 # 辅助损失增强一致性
# from .hypercorre8_cluster_t_add_norm import hypercorre_topk2 as hypercorre_topk_add_norm # metaformer
# from .hypercorre8_c_further import hypercorre_topk2 as hypercorre_topk_c_further # cffm c_further
# # test for ratio fusion
# # from .fdsf import FDSF,PagFM 
# from mmseg.models.backbones.pixel_decoder import PixelDecoder
# from mmseg.models.utils.detectron2_layers import ShapeSpec


# from mmseg.models.necks import EMCAD
from .utils.utils import save_cluster_labels
import time
from ..builder import build_loss
from torch.nn import functional as F


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2) #(bt,c,h*w) -> (bt,h*w,c)
        x = self.proj(x) #(bt,h*w,c) -> (bt,h*w,embed_dim)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='GN', num_groups=1)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        print(c1.shape, c2.shape, c3.shape, c4.shape)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        # print(torch.cuda.memory_allocated(0))

        return x


@HEADS.register_module()
class SegFormerHead_clips2_resize_1_8_hypercorrelation2_topk_ensemble4(BaseDecodeHead_clips):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips2_resize_1_8_hypercorrelation2_topk_ensemble4, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.deco1=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco2=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco3=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco4=small_decoder2(embedding_dim,256, self.num_classes)
        self.linear_qk = False
        self.hypercorre_module=hypercorre_topk2(dim=self.in_channels, backbone=self.backbone,linear_qk=self.linear_qk) # linear_qkv or cnn_qk

        reference_size="1_32"   ## choices: 1_32, 1_16
        if reference_size=="1_32":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)
        elif reference_size=="1_16":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True

    def forward(self, inputs, batch_size=None, num_clips=None, img_metas=None):
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        # print("backbone_out",c1.shape, c2.shape, c3.shape, c4.shape)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w)

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1]

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:]
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1]
        
        query_c2=query_c2.reshape(batch_size*(num_clips-1), -1, shape_c2[0], shape_c2[1])
        query_c3=query_c3.reshape(batch_size*(num_clips-1), -1, shape_c3[0], shape_c3[1])

        # query_c1=self.sr1(query_c1)
        query_c2=self.sr2(query_c2)
        query_c3=self.sr3(query_c3)

        # query_c1=query_c1.reshape(batch_size, (num_clips-1), -1, query_c1.shape[-2], query_c1.shape[-1])
        query_c2=query_c2.reshape(batch_size, (num_clips-1), -1, query_c2.shape[-2], query_c2.shape[-1])
        query_c3=query_c3.reshape(batch_size, (num_clips-1), -1, query_c3.shape[-2], query_c3.shape[-1])
        # query_c4=query_c4.reshape(batch_size, (num_clips-1), -1, query_c4.shape[-2], query_c4.shape[-1])

        query_frame=[query_c1, query_c2, query_c3, query_c4]
        supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]
        # supp_frame=[c1[-batch_size:].unsqueeze(1), c2[-batch_size:].unsqueeze(1), c3[-batch_size:].unsqueeze(1), c4[-batch_size:].unsqueeze(1)]
        # print([i.shape for i in query_frame])
        # print([i.shape for i in supp_frame])
        start_time11=time.time()
        atten, topk_mask=self.hypercorre_module(query_frame, supp_frame)
        # print(atten.shape, atten.max(), atten.min())
        # exit()
        atten=F.softmax(atten,dim=-1)

        start_time2=time.time()

        h2=int(h/2)
        w2=int(w/2)
        # h3,w3=shape_c3[-2], shape_c3[-1]
        _c2 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)
        _c2_split=_c2.reshape(batch_size, num_clips, -1, h2, w2)
        # _c_further=_c2[:,:-1].reshape(batch_size, num_clips-1, -1, h3*w3)
        _c3=self.sr1_feat(_c2)
        _c3=_c3.reshape(batch_size, num_clips, -1, _c3.shape[-2]*_c3.shape[-1]).transpose(-2,-1)
        # _c_further=_c3[:,:-1].reshape(batch_size, num_clips-1, _c2.shape[-2], _c2.shape[-1], -1)    ## batch_size, num_clips-1, _c2.shape[-2], _c2.shape[-1], c
        _c_further=_c3[:,:-1]        ## batch_size, num_clips-1, _c2.shape[-2]*_c2.shape[-1], c
        # print(_c_further.shape, topk_mask.shape, torch.unique(topk_mask.sum(2)))
        _c_further=_c_further[topk_mask].reshape(batch_size,num_clips-1,-1,_c_further.shape[-1])    ## batch_size, num_clips-1, s, c
        supp_feats=torch.matmul(atten,_c_further)
        supp_feats=supp_feats.transpose(-2,-1).reshape(batch_size, (num_clips-1), -1, h2,w2)
        supp_feats=(torch.chunk(supp_feats, (num_clips-1), dim=1))
        supp_feats=[ii.squeeze(1) for ii in supp_feats]
        supp_feats.append(_c2_split[:,-1])

        outs=supp_feats

        out1=resize(self.deco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out2=resize(self.deco2(outs[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out3=resize(self.deco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4(outs[3]+outs[2]+outs[1]+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4((outs[3]+outs[2]+outs[1])/3.0+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out4=resize(self.deco4((outs[0]+outs[1]+outs[2])/3.0+outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)

        output=torch.cat([x,out1,out2,out3,out4],dim=1)   ## b*(k+k)*124*h*w

        if not self.training:
            # return output.squeeze(1)
            # return torch.cat([x2,x3],1).mean(1)
            return out4.squeeze(1)
            # return out4.squeeze(1)+(out3.squeeze(1)+out2.squeeze(1)+out1.squeeze(1))/3
            # return F.softmax(torch.cat([out1,out2,out3,out4],1),dim=2).sum(1)
            # return torch.cat([out1,out2,out3,out4],1).mean(1)

        return output

# 原类别识别上更好，现在局部一致性更强，更加平滑，全局的linear对于局部的识别是有利的，但是linear又破坏了局部性质，时间信息上是有效的，
# 即使直接使用平均的方式效果也比不使用时间信息好
@HEADS.register_module()
class SegFormerHead_clips2_resize_1_8_static_dynamic_ensemble4(BaseDecodeHead_clips2): #decoders
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips2_resize_1_8_static_dynamic_ensemble4, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        self.lv = decoder_params['lv']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.fusion_decoder = EMCAD(channels=[embedding_dim,embedding_dim,embedding_dim,embedding_dim], kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation="relu6")

        self.deco1=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco2=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco3=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco4=small_decoder2(embedding_dim,256, self.num_classes)

        # self.hypercorre_module=hypercorre_topk21(dim=self.in_channels, backbone=self.backbone) # linear_qkv
        self.hypercorre_module=hypercorre_topk31(dim=self.in_channels, backbone=self.backbone)  #用decoder:

        reference_size="1_32"   ## choices: 1_32, 1_16
        if reference_size=="1_32":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)

        elif reference_size=="1_16":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True

    def forward(self, inputs, batch_size=None, num_clips=None, img_metas=None):
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x #(bt,c,h,w)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c41 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) #(bt,h*w,embed_dim) -> (bt,embed_dim,h,w)
        _c42 = resize(_c41, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c31 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c32 = resize(_c31, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c21 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c22 = resize(_c21, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c12 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c42, _c32, _c22, _c12], dim=1)) #(bt,embed_dim,h,w) 1/4

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w) #(b,t,c,h,w)

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1] #(b,c,h,w)

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:] #(h,w)
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1] #list of (b,t-1,c,h,w)
        
        # 这里不需要降低分辨率
        # query_c2=query_c2.reshape(batch_size*(num_clips-1), -1, shape_c2[0], shape_c2[1])
        # query_c3=query_c3.reshape(batch_size*(num_clips-1), -1, shape_c3[0], shape_c3[1])

        # # query_c1=self.sr1(query_c1)
        # query_c2=self.sr2(query_c2) #[b,t-1,c,1/8h,1/8w] -> [b,t-1,c,1/32h,1/32w]
        # query_c3=self.sr3(query_c3) #[b,t-1,c,1/16h,1/16w] -> [b,t-1,c,1/32h,1/32w]

        # # query_c1=query_c1.reshape(batch_size, (num_clips-1), -1, query_c1.shape[-2], query_c1.shape[-1])
        # query_c2=query_c2.reshape(batch_size, (num_clips-1), -1, query_c2.shape[-2], query_c2.shape[-1])
        # query_c3=query_c3.reshape(batch_size, (num_clips-1), -1, query_c3.shape[-2], query_c3.shape[-1])
        # # query_c4=query_c4.reshape(batch_size, (num_clips-1), -1, query_c4.shape[-2], query_c4.shape[-1])

        query_frame=[query_c1, query_c2, query_c3, query_c4]
        supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]

        v1 = _c12.reshape(batch_size, num_clips, -1, _c12.shape[-2], _c12.shape[-1])
        v2 = _c21.reshape(batch_size, num_clips, -1, _c21.shape[-2], _c21.shape[-1])
        v3 = _c31.reshape(batch_size, num_clips, -1, _c31.shape[-2], _c31.shape[-1])
        v4 = _c41.reshape(batch_size, num_clips, -1, _c41.shape[-2], _c41.shape[-1])
        v_frame = [v1[:,:-1], v2[:,:-1], v3[:,:-1], v4[:,:-1]]

        # supp_frame=[c1[-batch_size:].unsqueeze(1), c2[-batch_size:].unsqueeze(1), c3[-batch_size:].unsqueeze(1), c4[-batch_size:].unsqueeze(1)]
        # print([i.shape for i in query_frame])
        # print([i.shape for i in supp_frame])
        start_time11=time.time()
        # Q: supp_frame (b,1,c,h,w) k:query_frame (b,t-1,c,h,w) v:

        supp_feats=self.hypercorre_module(query_frame, supp_frame,v_frame,self.lv) #[[b,c,h,w]]
        # print(atten.shape, atten.max(), atten.min())
        # exit()
        # atten=F.softmax(atten,dim=-1)

        start_time2=time.time()

        # h2=int(h/2)
        # w2=int(w/2)
        # h3,w3=shape_c3[-2], shape_c3[-1]
        # _c2 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)  #1/8
        # _c2_split=_c2.reshape(batch_size, num_clips, -1, h2, w2)  # 1/8
        # supp_feats.append(_c2_split[:,-1])
        _c = _c.reshape(batch_size, num_clips, -1, h, w)
        supp_feats.append(_c[:,-1])

        outs=supp_feats
        # print("outs",outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape)

        outs = self.fusion_decoder(outs[0],[outs[1],outs[2],outs[3]])

        out1=resize(self.deco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/32 ,b,c,h,w
        out2=resize(self.deco2(outs[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/16
        out3=resize(self.deco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/8
        out4 = resize(self.deco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/4
        # out4=resize(self.deco4(outs[3]+outs[2]+outs[1]+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4((outs[3]+outs[2]+outs[1])/3.0+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
       
        # out_41 = resize(outs[0], size=(h//2, w//2),mode='bilinear',align_corners=False) 
        # out_42 = resize(outs[1], size=(h//2, w//2),mode='bilinear',align_corners=False) 
        # out_43 = resize(outs[2], size=(h//2, w//2),mode='bilinear',align_corners=False)
        # # # print("out",out_41.shape, out_42.shape, out_43.shape,outs[3].shape)
        # out4=resize(self.deco4((out_41+out_42+out_43)/3.0+outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/4 这里concat用1x1应该好一点

        output=torch.cat([x,out1,out2,out3,out4],dim=1)   ## b*(k+k)*124*h*w
        # 问题：1. 受低分辨率对齐的损失影响 2. 高低分辨率直接相加结果

        if not self.training:
            # return output.squeeze(1)
            # return torch.cat([x2,x3],1).mean(1)
            return out4.squeeze(1)
            # return out4.squeeze(1)+(out3.squeeze(1)+out2.squeeze(1)+out1.squeeze(1))/3
            # return F.softmax(torch.cat([out1,out2,out3,out4],1),dim=2).sum(1)
            # return torch.cat([out1,out2,out3,out4],1).mean(1)

        return output



@HEADS.register_module()
class SegFormerHead_clips2_resize_1_8_static_dynamic_ensemble4_2(BaseDecodeHead_clips2):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet all cnn
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips2_resize_1_8_static_dynamic_ensemble4_2, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        self.lv = decoder_params['lv']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # work_dirs_all_cnn_hypper_4_has_decoder_1_8
        # self.fusion_decoder = EMCAD(channels=self.in_channels[::-1], kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation="relu6",layer=4)
        # work_dirs_all_cnn_hypper_4_has_decoder_1_4
        self.fusion_decoder = EMCAD(channels=[embedding_dim,embedding_dim,embedding_dim,embedding_dim], kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation="relu6",layer=4)

        # self.out_decoder = nn.Conv2d(c2_in_channels, embedding_dim, kernel_size=1)

        self.deco1=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco2=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco3=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco4=small_decoder2(embedding_dim,256, self.num_classes)

        self.linear_v = False # false效果贼差
        # self.hypercorre_module=hypercorre_topk21(dim=self.in_channels, backbone=self.backbone) #decoder
        self.hypercorre_module=hypercorre_topk41(dim=self.in_channels, backbone=self.backbone,linear_v=self.linear_v,embedding_dim=embedding_dim)  #cnn-all, each frame for target, each ratio for eacher

        reference_size="1_32"   ## choices: 1_32, 1_16
        if reference_size=="1_32":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)

        elif reference_size=="1_16":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True

    def forward(self, inputs, batch_size=None, num_clips=None, img_metas=None):
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x #(bt,c,h,w)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c41 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) #(bt,h*w,embed_dim) -> (bt,embed_dim,h,w)
        _c42 = resize(_c41, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c31 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c32 = resize(_c31, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c21 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c22 = resize(_c21, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c12 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c42, _c32, _c22, _c12], dim=1)) #(bt,embed_dim,h,w) 1/4

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w) #(b,t,c,h,w)

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1] #(b,c,h,w)

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:] #(h,w)
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1] #list of (b,t-1,c,h,w)
        
        # 这里不需要降低分辨率
        # query_c2=query_c2.reshape(batch_size*(num_clips-1), -1, shape_c2[0], shape_c2[1])
        # query_c3=query_c3.reshape(batch_size*(num_clips-1), -1, shape_c3[0], shape_c3[1])

        # # query_c1=self.sr1(query_c1)
        # query_c2=self.sr2(query_c2) #[b,t-1,c,1/8h,1/8w] -> [b,t-1,c,1/32h,1/32w]
        # query_c3=self.sr3(query_c3) #[b,t-1,c,1/16h,1/16w] -> [b,t-1,c,1/32h,1/32w]

        # # query_c1=query_c1.reshape(batch_size, (num_clips-1), -1, query_c1.shape[-2], query_c1.shape[-1])
        # query_c2=query_c2.reshape(batch_size, (num_clips-1), -1, query_c2.shape[-2], query_c2.shape[-1])
        # query_c3=query_c3.reshape(batch_size, (num_clips-1), -1, query_c3.shape[-2], query_c3.shape[-1])
        # # query_c4=query_c4.reshape(batch_size, (num_clips-1), -1, query_c4.shape[-2], query_c4.shape[-1])
        if not self.linear_v:
            query_frame=[query_c1, query_c2, query_c3, query_c4]
            supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]
            v_frame = [query_c1, query_c2, query_c3, query_c4]

        else:     
            v1 = _c12.reshape(batch_size, num_clips, -1, _c12.shape[-2], _c12.shape[-1])
            v2 = _c21.reshape(batch_size, num_clips, -1, _c21.shape[-2], _c21.shape[-1])
            v3 = _c31.reshape(batch_size, num_clips, -1, _c31.shape[-2], _c31.shape[-1])
            v4 = _c41.reshape(batch_size, num_clips, -1, _c41.shape[-2], _c41.shape[-1])
            v_frame = [v1[:,:-1], v2[:,:-1], v3[:,:-1], v4[:,:-1]]

        # supp_frame=[c1[-batch_size:].unsqueeze(1), c2[-batch_size:].unsqueeze(1), c3[-batch_size:].unsqueeze(1), c4[-batch_size:].unsqueeze(1)]
        # print([i.shape for i in query_frame])
        # print([i.shape for i in supp_frame])
        start_time11=time.time()
        # Q: supp_frame (b,1,c,h,w) k:query_frame (b,t-1,c,h,w) v:

        supp_feats=self.hypercorre_module(query_frame, supp_frame,v_frame,self.lv) #[[b*t,c,h/32,w/32],[b*t,c,h/16,w/16],[b*t,c,h/8,w/8]]
        # print(atten.shape, atten.max(), atten.min())
        # exit()
        # atten=F.softmax(atten,dim=-1)

        start_time2=time.time()

        # h2=int(h/2)
        # w2=int(w/2)
        # # h3,w3=shape_c3[-2], shape_c3[-1]
        # _c2 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)  #1/8
        # _c2_split=_c2.reshape(batch_size, num_clips, -1, h2, w2)  # 1/8
        # supp_feats.append(_c2_split[:,-1])
        _c = _c.reshape(batch_size, num_clips, -1, h, w)
        supp_feats.append(_c[:,-1])
        outs=supp_feats
        outs[0] = outs[0].reshape(batch_size, num_clips-1, -1, outs[0].shape[-2], outs[0].shape[-1]).sum(1) / (num_clips-1) # 1/32
        outs[1] = outs[1].reshape(batch_size, num_clips-1, -1, outs[1].shape[-2], outs[1].shape[-1]).sum(1) / (num_clips-1) # 1/16
        outs[2] = outs[2].reshape(batch_size, num_clips-1, -1, outs[2].shape[-2], outs[2].shape[-1]).sum(1) / (num_clips-1) # 1/8
        # print("outs",len(outs),outs[0].shape,outs[1].shape,outs[2].shape)

        out = self.fusion_decoder(outs[0],[outs[1],outs[2],outs[3]])
        # supp_feats_out = out[-1].reshape(batch_size, num_clips-1, -1, h, w) # 1/32
        # outs_2 = supp_feats_out
        # outs_2 = self.out_decoder(supp_feats_out).reshape(batch_size, num_clips-1, -1, h2, w2) # 1/8 
        # print("reshape",outs_2.shape)
        # outs_2=(torch.chunk(outs_2, (num_clips-1), dim=1))
        # outs_2=[ii.squeeze(1) for ii in outs_2]
        # outs_2.append(_c2_split[:,-1])

        out1=resize(self.deco1(out[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/32 ,b,c,h,w
        out2=resize(self.deco2(out[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/16
        out3=resize(self.deco3(out[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/8
        # out4=resize(self.deco4(outs_2[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/4
        # out4=resize(self.deco4(outs[3]+outs[2]+outs[1]+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4((outs[3]+outs[2]+outs[1])/3.0+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
       
        out_41 = resize(outs[0], size=(h, w),mode='bilinear',align_corners=False) 
        out_42 = resize(outs[1], size=(h, w),mode='bilinear',align_corners=False) 
        out_43 = resize(outs[2], size=(h, w),mode='bilinear',align_corners=False)
        # # print("out",out_41.shape, out_42.shape, out_43.shape,outs[3].shape)
        out4=resize(self.deco4((out_41+out_42+out_43)/3.0+_c[:,-1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/4 这里concat用1x1应该好一点
        output=torch.cat([x,out1,out2,out3,out4],dim=1)   ## b*(k+k)*124*h*w

        if not self.training:
            # return output.squeeze(1)
            # return torch.cat([x2,x3],1).mean(1)
            return out4.squeeze(1)
            # return out4.squeeze(1)+(out3.squeeze(1)+out2.squeeze(1)+out1.squeeze(1))/3
            # return F.softmax(torch.cat([out1,out2,out3,out4],1),dim=2).sum(1)
            # return torch.cat([out1,out2,out3,out4],1).mean(1)

        return output

@HEADS.register_module()
class SegFormerHead_clips2_resize_1_8_static_dynamic_ensemble4_3(BaseDecodeHead_clips2): # CAT_blocks for cross and refiner time
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips2_resize_1_8_static_dynamic_ensemble4_3, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        self.lv = decoder_params['lv']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # self.fusion_decoder = EMCAD(channels=[embedding_dim,embedding_dim,embedding_dim,embedding_dim], kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation="relu6")

        self.deco1=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco2=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco3=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco4=small_decoder2(embedding_dim,256, self.num_classes)

        self.hypercorre_module=hypercorre_topk51(dim=self.in_channels, num_layers=1,t=3,time_decoder_layer=3,embedding_dim=embedding_dim) # linear_qkv
        # self.hypercorre_module=hypercorre_topk31(dim=self.in_channels, backbone=self.backbone)  #cnn-linear,没用decoder:0.5,用decoder:
        # 多尺度特征融合 MSDA
        #1. FDSF
        # self.fusion1 = FDSF(embedding_dim,embedding_dim)
        # self.fusion2 = FDSF(embedding_dim,embedding_dim)
        # self.fusion3 = FDSF(embedding_dim,embedding_dim)
        
        #2. MDSA
        self.keys = ['l4', 'l8', 'l16', 'l32']
        input_shape = collections.OrderedDict([("l4", ShapeSpec(channels=embedding_dim, stride=4)),("l8",ShapeSpec(channels=embedding_dim, stride=8)),
                                                  ("l16", ShapeSpec(channels=embedding_dim, stride=16)),
                                                  ("l32", ShapeSpec(channels=embedding_dim, stride=32))])
        
        self.pixel_decoder = PixelDecoder(  # num_point = 4
                    input_shape=input_shape,
                    transformer_dropout=0.0,
                    transformer_nheads=8,
                    transformer_dim_feedforward=1024,
                    transformer_enc_layers=6, #6 
                    conv_dim=256,
                    mask_dim=256,
                    norm='GN',
                    transformer_in_features=['l8', 'l16', 'l32'],
                    common_stride=4,
                    temporal_attn_patches_per_dim=8)

        # PAGFM
        # self.fusion1 = PagFM(embedding_dim,embedding_dim)
        # self.fusion2 = PagFM(embedding_dim,embedding_dim)
        # self.fusion3 = PagFM(embedding_dim,embedding_dim)
    
        reference_size="1_32"   ## choices: 1_32, 1_16
        if reference_size=="1_32":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)

        elif reference_size=="1_16":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True

    def forward(self, inputs, batch_size=None, num_clips=None, img_metas=None):
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x #(bt,c,h,w)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c41 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) #(bt,h*w,embed_dim) -> (bt,embed_dim,h,w)
        _c42 = resize(_c41, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c31 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c32 = resize(_c31, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c21 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c22 = resize(_c21, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c12 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c42, _c32, _c22, _c12], dim=1)) #(bt,embed_dim,h,w) 1/4

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w) #(b,t,c,h,w)

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1] #(b,c,h,w)

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:] #(h,w)
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1] #list of (b,t-1,c,h,w)
        
        query_frame=[query_c1, query_c2, query_c3, query_c4]
        supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]



        start_time11=time.time()
        # Q: supp_frame (b,1,c,h,w) k:query_frame (b,t-1,c,h,w) v:

        supp_feats=self.hypercorre_module(query_frame, supp_frame) #[[b,c,h,w]]
        # print(atten.shape, atten.max(), atten.min())
        # exit()
        # atten=F.softmax(atten,dim=-1)

        start_time2=time.time()
        # 多尺度交互 ,1/32具有充分的语义，1/8具有充分的细节
        # 1. 全融合： MSDA
        # 2. 逐步融合： FDSF

        # h2=int(h/2)
        # w2=int(w/2)
        # h3,w3=shape_c3[-2], shape_c3[-1]
        # _c2 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)  #1/8 这里必须要下采样吗？，当向上采样的时候，就是需要浅层得到的结果，
        # _c2_split=_c2.reshape(batch_size, num_clips, -1, h2, w2)  # 1/8
        # supp_feats.append(_c2_split[:,-1])

        _c = _c.reshape(batch_size, num_clips, -1, h, w)
        supp_feats.append(_c[:,-1])

        outs=supp_feats

        # 非迭代 FDSF and  PAGFM
        # out0 = outs[0]
        # out1 = self.fusion1(outs[1],outs[0])
        # out2 = self.fusion2(outs[2],outs[1])
        # out3 = self.fusion3(outs[3],outs[2])
        # outs = [out0,out1,out2,out3]
        
        #MSDA
        multi_scale_features =  collections.OrderedDict()
        multi_scale_features['l32'] = outs[0]
        multi_scale_features['l16'] = outs[1]
        multi_scale_features['l8'] = outs[2]
        multi_scale_features['l4'] = outs[3]
        # print("multi_scale_features",multi_scale_features['l32'].shape,multi_scale_features['l16'].shape,multi_scale_features['l8'].shape,multi_scale_features['l4'].shape)
        outs = self.pixel_decoder(multi_scale_features)

        # print("outs",outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape)

        # outs = self.fusion_decoder(outs[0],[outs[1],outs[2],outs[3]])
        out1=resize(self.deco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/32 ,b,c,h,w
        out2=resize(self.deco2(outs[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/16
        out3=resize(self.deco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/8
        out4 = resize(self.deco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/4
        # out4=resize(self.deco4(outs[3]+outs[2]+outs[1]+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4((outs[3]+outs[2]+outs[1])/3.0+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
       
        # out_41 = resize(outs[0], size=(h//2, w//2),mode='bilinear',align_corners=False) 
        # out_42 = resize(outs[1], size=(h//2, w//2),mode='bilinear',align_corners=False) 
        # out_43 = resize(outs[2], size=(h//2, w//2),mode='bilinear',align_corners=False)
        # # # print("out",out_41.shape, out_42.shape, out_43.shape,outs[3].shape)
        # out4=resize(self.deco4((out_41+out_42+out_43)/3.0+outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/4 这里concat用1x1应该好一点

        output=torch.cat([x,out1,out2,out3,out4],dim=1)   ## b*(k+k)*124*h*w
        # 问题：1. 受低分辨率对齐的损失影响 2. 高低分辨率直接相加结果

        if not self.training:
            # return output.squeeze(1)
            # return torch.cat([x2,x3],1).mean(1)
            return out4.squeeze(1)
            # return out4.squeeze(1)+(out3.squeeze(1)+out2.squeeze(1)+out1.squeeze(1))/3
            # return F.softmax(torch.cat([out1,out2,out3,out4],1),dim=2).sum(1)
            # return torch.cat([out1,out2,out3,out4],1).mean(1)

        return output


@HEADS.register_module()
class SegFormerHead_clips2_resize_1_8_CAT_SegDeformer_ensemble4(BaseDecodeHead_clips2):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips2_resize_1_8_CAT_SegDeformer_ensemble4, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.deco1=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco2=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco3=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco4=small_decoder2(embedding_dim,256, self.num_classes)
        #CAT blocks and Segdeformer for time and ratio fusion all
        self.cross_method = kwargs['cross_method'] # [Focal,CAT]
        self.ratio_fusio = True
        self.num_layer=2
        print("-------in model: cross_method:",self.cross_method,"using PagFM for ratio_fusio:",self.ratio_fusio) 
        self.hypercorre_module=hypercorre_topk61(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,cross_method=self.cross_method,ratio_fusio=self.ratio_fusio) # linear_qkv or cnn_qk
        
        #CAT_blk and PAGFM for ratio fusion and Segdeformer  for time fusion
        # self.hypercorre_module=hypercorre_topk71(dim=self.in_channels, num_layers=1,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes) # linear_qkv or cnn_qk

        reference_size="1_32"   ## choices: 1_32, 1_16
        if reference_size=="1_32":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)
        elif reference_size=="1_16":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True

    def forward(self, inputs, batch_size=None, num_clips=None, img_metas=None):
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x #(bt,c,h,w)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c41 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) #(bt,h*w,embed_dim) -> (bt,embed_dim,h,w)
        _c42 = resize(_c41, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c31 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c32 = resize(_c31, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c21 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c22 = resize(_c21, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c12 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c42, _c32, _c22, _c12], dim=1)) #(bt,embed_dim,h,w) 1/4

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w) #(b,t,c,h,w)

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1] #(b,c,h,w)

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:] #(h,w)
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1] #list of (b,t-1,c,h,w)
        
        query_frame=[query_c1, query_c2, query_c3, query_c4]
        supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]


        #from cffm 传入_c
        # h2=int(h/2)
        # w2=int(w/2)
        # _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False) #降采样
        # _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2)
        # query_frame=_c_further[:,:-1]
        # supp_frame=_c_further[:,-1:]



        start_time11=time.time()
        # Q: supp_frame (b,1,c,h,w) k:query_frame (b,t-1,c,h,w) v:

        start_time2=time.time()
        h2=int(h/2)
        w2=int(w/2)
        h3,w3=shape_c3[-2], shape_c3[-1]
        _c2 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)  #1/8 这里必须要下采样吗？，当向上采样的时候，就是需要浅层得到的结果，
        _c2_split=_c2.reshape(batch_size, num_clips, -1, h2, w2)  # 1/8


        supp_feats,out_cls_mid=self.hypercorre_module(query_frame, supp_frame) #[t [b,c,h,w]]
        # print("out_cls_mid",out_cls_mid.shape)
        # print(atten.shape, atten.max(), atten.min())
        # exit()
        # atten=F.softmax(atten,dim=-1)

        if len(supp_feats) < num_clips: # hypper6
            supp_feats.append(_c2_split[:,-1])

        # _c = _c.reshape(batch_size, num_clips, -1, h, w)
        # supp_feats.append(_c[:,-1])
        
        outs=supp_feats
        # print("outs",outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape)

        # outs = self.fusion_decoder(outs[0],[outs[1],outs[2],outs[3]])

        out1=resize(self.deco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/32 ,b,c,h,w
        out2=resize(self.deco2(outs[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/16
        out3=resize(self.deco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/8
        # out4 = resize(self.deco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/4
        # out4=resize(self.deco4(outs[3]+outs[2]+outs[1]+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4((outs[3]+outs[2]+outs[1])/3.0+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
       
        # out_41 = resize(outs[0], size=(h//2, w//2),mode='bilinear',align_corners=False) 
        # out_42 = resize(outs[1], size=(h//2, w//2),mode='bilinear',align_corners=False) 
        # out_43 = resize(outs[2], size=(h//2, w//2),mode='bilinear',align_corners=False)
        # # # print("out",out_41.shape, out_42.shape, out_43.shape,outs[3].shape)
        if self.cross_method == 'CAT' or self.cross_method == 'Focal_CAT': # for target frame
            out4=resize(self.deco4((outs[0]+outs[1]+outs[2])/3.0+outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        elif self.cross_method == 'Focal': # for all frame
            out4=resize(self.deco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/4
            

        out_cls_mid = [resize(out_cls_mid[:,i], size=(h, w),mode='bilinear',align_corners=False) for i in range(num_clips-1 if (self.cross_method == 'CAT' or self.cross_method == 'Focal_CAT') else num_clips)] #[b,t,c,h,w]
        out_cls_mid = torch.stack(out_cls_mid, dim=1) #(b,t,c,h,w)
        output=torch.cat([x,out1,out2,out3,out4,out_cls_mid],dim=1)   ## b*(k+2k)*124*h*w
        
        if not self.training:
            # return output.squeeze(1)
            # return torch.cat([x2,x3],1).mean(1)
            return out4.squeeze(1)
            # return out4.squeeze(1)+(out3.squeeze(1)+out2.squeeze(1)+out1.squeeze(1))/3
            # return F.softmax(torch.cat([out1,out2,out3,out4],1),dim=2).sum(1)
            # return torch.cat([out1,out2,out3,out4],1).mean(1)

        return output



@HEADS.register_module()
class SegFormerHead_clips2_resize_1_8_CAT_SegDeformer_MaskGuided_ensemble4(BaseDecodeHead_clips2):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips2_resize_1_8_CAT_SegDeformer_MaskGuided_ensemble4, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1) 

        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.deco1=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco2=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco3=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco4=small_decoder2(embedding_dim,256, self.num_classes)
        #CAT blocks and Segdeformer for time and ratio fusion all
        self.cross_method = kwargs['cross_method'] # [Focal,CAT]
        print("-------in model: cross_method:",self.cross_method) 
        seg_mask_use =   kwargs['seg_mask_use']            #['cluster_guide','cluster_reduce_cmp','cluster_Agent','mask_guide','top-k_mask']
        self.hypercorre_module=hypercorre_topk81(dim=self.in_channels, num_layers=1,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,cross_method=self.cross_method,seg_mask_use=seg_mask_use) # linear_qkv or cnn_qk
        
        #CAT_blk and PAGFM for ratio fusion and Segdeformer  for time fusion
        # self.hypercorre_module=hypercorre_topk71(dim=self.in_channels, num_layers=1,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes) # linear_qkv or cnn_qk

        reference_size="1_32"   ## choices: 1_32, 1_16
        if reference_size=="1_32":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)
        elif reference_size=="1_16":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True

    def forward(self, inputs, batch_size=None, num_clips=None, img_metas=None):
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x #(bt,c,h,w)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c41 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) #(bt,h*w,embed_dim) -> (bt,embed_dim,h,w)
        _c42 = resize(_c41, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c31 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c32 = resize(_c31, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c21 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c22 = resize(_c21, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c12 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c42, _c32, _c22, _c12], dim=1)) #(bt,embed_dim,h,w) 1/4

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        _x = self.linear_pred(x)
        x = _x.reshape(batch_size, num_clips, -1, h, w) #(b,t,c,h,w) # here get mask 

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1] #(b,c,h,w)

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:] #(h,w)
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1] #list of (b,t-1,c,h,w)
        
        query_frame=[query_c1, query_c2, query_c3, query_c4]
        supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]


        #from cffm 传入_c
        # h2=int(h/2)
        # w2=int(w/2)
        # _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False) #降采样
        # _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2)
        # query_frame=_c_further[:,:-1]
        # supp_frame=_c_further[:,-1:]



        start_time11=time.time()
        # Q: supp_frame (b,1,c,h,w) k:query_frame (b,t-1,c,h,w) v:

        start_time2=time.time()
        h2=int(h/2)
        w2=int(w/2)
        h3,w3=shape_c3[-2], shape_c3[-1]
        _c2 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)  #1/8 这里必须要下采样吗？，当向上采样的时候，就是需要浅层得到的结果，
        _c2_split=_c2.reshape(batch_size, num_clips, -1, h2, w2)  # 1/8

        seg_x = resize(_x, size=(h2,w2),mode='bilinear',align_corners=False).reshape(batch_size, num_clips, -1, h2, w2)  #1/8
        supp_feats,out_cls_mid=self.hypercorre_module(query_frame, supp_frame, seg_mask = seg_x) #[t [b,c,h,w]]
        # print("out_cls_mid",out_cls_mid.shape)
        # print(atten.shape, atten.max(), atten.min())
        # exit()
        # atten=F.softmax(atten,dim=-1)

        if len(supp_feats) < num_clips: # hypper6
            supp_feats.append(_c2_split[:,-1])

        # _c = _c.reshape(batch_size, num_clips, -1, h, w)
        # supp_feats.append(_c[:,-1])
        
        outs=supp_feats
        # print("outs",outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape)

        # outs = self.fusion_decoder(outs[0],[outs[1],outs[2],outs[3]])

        out1=resize(self.deco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/32 ,b,c,h,w
        out2=resize(self.deco2(outs[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) #1/16
        out3=resize(self.deco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/8
        # out4 = resize(self.deco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/4
        # out4=resize(self.deco4(outs[3]+outs[2]+outs[1]+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4((outs[3]+outs[2]+outs[1])/3.0+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
       
        # out_41 = resize(outs[0], size=(h//2, w//2),mode='bilinear',align_corners=False) 
        # out_42 = resize(outs[1], size=(h//2, w//2),mode='bilinear',align_corners=False) 
        # out_43 = resize(outs[2], size=(h//2, w//2),mode='bilinear',align_corners=False)
        # # # print("out",out_41.shape, out_42.shape, out_43.shape,outs[3].shape)
        if self.cross_method == 'CAT': # for target frame
            out4=resize(self.deco4((outs[0]+outs[1]+outs[2])/3.0+outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        elif self.cross_method == 'Focal': # for all frame
            out4=resize(self.deco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1) # 1/4
            

        out_cls_mid = [resize(out_cls_mid[:,i], size=(h, w),mode='bilinear',align_corners=False) for i in range(num_clips-1 if self.cross_method == 'CAT' else num_clips)] #[b,t,c,h,w]
        out_cls_mid = torch.stack(out_cls_mid, dim=1) #(b,t,c,h,w)
        output=torch.cat([x,out1,out2,out3,out4,out_cls_mid],dim=1)   ## b*(k+2k)*124*h*w
        
        if not self.training:
            # return output.squeeze(1)
            # return torch.cat([x2,x3],1).mean(1)
            return out4.squeeze(1)
            # return out4.squeeze(1)+(out3.squeeze(1)+out2.squeeze(1)+out1.squeeze(1))/3
            # return F.softmax(torch.cat([out1,out2,out3,out4],1),dim=2).sum(1)
            # return torch.cat([out1,out2,out3,out4],1).mean(1)

        return output


@HEADS.register_module()
class SegFormerHead_clips2_resize_1_8_Cluster_SegDeformer_ensemble4(BaseDecodeHead_clips2):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        print("in SegFormerHead_clips2_resize_1_8_Cluster_SegDeformer_ensemble4")
        super(SegFormerHead_clips2_resize_1_8_Cluster_SegDeformer_ensemble4, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        cityscape = kwargs['cityscape']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)


        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        #CAT blocks and Segdeformer for time and ratio fusion all
        self.cross_method = kwargs['cross_method'] # [Focal,CAT]
        self.num_clusters = kwargs['num_cluster']
        self.cluster_with_t = kwargs['cluster_with_t']
        self.need_segdeformer = kwargs['need_segdeformer'] #测试在聚类中是否需要特定的decoder结构
        self.aux_loss_decode = kwargs['aux_loss_decode']
        backbone = kwargs['backbone']
        if self.aux_loss_decode:
            self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
            
        self.ratio_fusio =  False  ##"sub" ❌  "freq_fuse" ❌ 'GAU' ❌
        self.num_layer=2
        self.test_only_decoder = False # True:38.58
        if self.test_only_decoder:
            self.linear_pred1 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        else:
            self.linear_pred1 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)

        print("-------in model: cross_method:",self.cross_method,"using PagFM for ratio_fusio:",self.ratio_fusio,"cluster_with_t",self.cluster_with_t,"need_segdeformer",self.need_segdeformer) 
        #cluster(paca-vit)
        print("---------self.num_class:",self.num_classes)

        self.hypercorre_module=hypercorre_topk91(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
                                                 cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters,cluster_with_t=self.cluster_with_t,backbone=backbone,
                                                 cityscape=cityscape,need_segdeformer = self.need_segdeformer) # linear_qkv or cnn_qk
        
        # cluster(paca-vit)
        
        # self.hypercorre_module=hypercorre_topk92(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters,cluster_with_t=self.cluster_with_t,aux_loss_decode=self.aux_loss_decode) # linear_qkv or cnn_qk
        
        # # cluster(cluster former)
        # self.hypercorre_module=hypercorre_topk101(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters) # linear_qkv or cnn_qk
        
        # cluster(cluster former)
        # self.hypercorre_module=hypercorre_topk111(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters) # linear_qkv or cnn_qk
        
        # self.hypercorre_module=hypercorre_topk121(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters,cluster_with_t=self.cluster_with_t) # linear_qkv or cnn_qk

        # self.hypercorre_module=hypercorre_topk131(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters,cluster_with_t=self.cluster_with_t,need_segdeformer=self.need_segdeformer) # linear_qkv or cnn_qk

        # meta former
        # self.hypercorre_module=hypercorre_topk_add_norm(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters,cluster_with_t=self.cluster_with_t) # linear_qkv or cnn_qk
        

        # reference_size="1_32"   ## choices: 1_32, 1_16
        # if reference_size=="1_32":
        #     # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
        #     self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
        #     self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
        #     self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)
        # elif reference_size=="1_16":
        #     # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
        #     self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
        #     self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
        #     self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True

    def forward(self, inputs, batch_size=None, num_clips=None, img_metas=None):
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x #(bt,c,h,w)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c41 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) #(bt,h*w,embed_dim) -> (bt,embed_dim,h,w)
        _c42 = resize(_c41, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c31 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c32 = resize(_c31, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c21 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c22 = resize(_c21, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c12 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c42, _c32, _c22, _c12], dim=1)) #(bt,embed_dim,h,w) 1/4

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w) #(b,t,c,h,w)

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1] #(b,c,h,w)

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:] #(h,w)
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1] #list of (b,t-1,c,h,w)
        supp1,supp2,supp3,supp4=c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:] #list of (b,1,c,h,w)
        query_frame=[query_c1, query_c2, query_c3, query_c4]
        supp_frame=[supp1, supp2, supp3, supp4]

        h2=int(h/2)
        w2=int(w/2)
        _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False) #降采样
        _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2)
        
        # query_frame=_c_further[:,:-1]
        # supp_frame=_c_further[:,-1:] #这里包含了更加细节的信息，相较于直接下采样再融合的情况来说
        # supp_feats,out_cls_mid=self.hypercorre_module(query_frame, supp_frame,img_metas = img_metas) # target: [2 [b,c,h,w]] cluster_centers:[b,t,num_clusters,c,h,w]
        
        out_cls_mid = None

        if self.need_segdeformer:
            supp_feats,out_cls_mid,cluster_centers,mem_out=self.hypercorre_module(query_frame, supp_frame,img_metas = img_metas) # target: [2 [b,c,h,w]] cluster_centers:[b,t,num_clusters,c,h,w]
        else:
            supp_feats,cluster_centers=self.hypercorre_module(query_frame, supp_frame,img_metas = img_metas)

        # 测试只要decoder结果
        if self.test_only_decoder:
            x2 = resize(self.linear_pred1(supp_feats[0]), size=(h,w),mode='bilinear',align_corners=False)
            x2 = x2.unsqueeze(1)
        else:
            _c_further2=torch.cat([_c_further[:,-1], supp_feats[0]],1)
            x2 = self.dropout(_c_further2)
            x2 = self.linear_pred1(x2)
            x2=resize(x2, size=(h,w),mode='bilinear',align_corners=False)
            x2=x2.unsqueeze(1)

        if not self.training or not self.aux_loss_decode:
            if self.need_segdeformer:
                x3 = resize(out_cls_mid, size=(h,w),mode='bilinear',align_corners=False).unsqueeze(1)
                # print("x2",x2.shape, "x3",x3.shape,x.shape)
                output=torch.cat([x,x2,x3],dim=1)   ## b*(k+2)*124*h*w
            else:   
                output=torch.cat([x,x2],dim=1)   ## b*(k+2)*124*h*w
        else:
            if self.need_segdeformer:
                x4 = [resize(self.linear_pred2(mem_out[i]), size=(h,w),mode='bilinear',align_corners=False).unsqueeze(1) for i in range(len(mem_out))] #[b,t,c,h,w]
                x4 = torch.cat(x4, dim=1) #(b,t,c,h,w)
                x3 = resize(out_cls_mid, size=(h,w),mode='bilinear',align_corners=False).unsqueeze(1)
                # print("x2",x2.shape, "x3",x3.shape,x.shape)
                output=torch.cat([x,x2,x3,x4],dim=1)   ## b*(k+2)*124*h*w
            else:   
                output=torch.cat([x,x2],dim=1)   ## b*(k+2)*124*h*w

        if not self.training:
            return x2.squeeze(1)
        return output,cluster_centers
        # return output



# 解决cffm与当前区别，看看是否会导致小目标分割问题
@HEADS.register_module()
class SegFormerHead_clips2_resize_1_8_Cluster_SegDeformer_C_Futher_ensemble4(BaseDecodeHead_clips2):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips2_resize_1_8_Cluster_SegDeformer_C_Futher_ensemble4, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred1 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)


        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        #CAT blocks and Segdeformer for time and ratio fusion all
        self.cross_method = kwargs['cross_method'] # [Focal,CAT]
        self.num_clusters = kwargs['num_cluster']
        self.cluster_with_t = kwargs['cluster_with_t']
        self.need_segdeformer = kwargs['need_segdeformer'] #测试在聚类中是否需要特定的decoder结构
        self.aux_loss_decode = kwargs['aux_loss_decode']
        if self.aux_loss_decode:
            self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
            
        self.ratio_fusio =  False  ##"sub" 
        self.num_layer=2
        print("-------in model: cross_method:",self.cross_method,"using PagFM for ratio_fusio:",self.ratio_fusio,"cluster_with_t",self.cluster_with_t,"need_segdeformer",self.need_segdeformer) 
        #cluster(paca-vit)

        self.hypercorre_module=hypercorre_topk_c_further(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
                                                 cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters,cluster_with_t=self.cluster_with_t) # linear_qkv or cnn_qk
        
        # cluster(paca-vit)
        
        # self.hypercorre_module=hypercorre_topk92(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters,cluster_with_t=self.cluster_with_t,aux_loss_decode=self.aux_loss_decode) # linear_qkv or cnn_qk
        
        # # cluster(cluster former)
        # self.hypercorre_module=hypercorre_topk101(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters) # linear_qkv or cnn_qk
        
        # cluster(cluster former)
        # self.hypercorre_module=hypercorre_topk111(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters) # linear_qkv or cnn_qk
        
        # self.hypercorre_module=hypercorre_topk121(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters,cluster_with_t=self.cluster_with_t) # linear_qkv or cnn_qk

        # self.hypercorre_module=hypercorre_topk131(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters,cluster_with_t=self.cluster_with_t,need_segdeformer=self.need_segdeformer) # linear_qkv or cnn_qk

        # meta former
        # self.hypercorre_module=hypercorre_topk_add_norm(dim=self.in_channels, num_layers=self.num_layer,t=3,time_decoder_layer=3,embedding_dim=embedding_dim,num_classes=self.num_classes,
        #                                          cross_method=self.cross_method,ratio_fusio=self.ratio_fusio,num_clusters=self.num_clusters,cluster_with_t=self.cluster_with_t) # linear_qkv or cnn_qk
        
        reference_size="1_32"   ## choices: 1_32, 1_16
        if reference_size=="1_32":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)
        elif reference_size=="1_16":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True

    def forward(self, inputs, batch_size=None, num_clips=None, img_metas=None):
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x #(bt,c,h,w)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c41 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) #(bt,h*w,embed_dim) -> (bt,embed_dim,h,w)
        _c42 = resize(_c41, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c31 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c32 = resize(_c31, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c21 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c22 = resize(_c21, size=c1.size()[2:],mode='bilinear',align_corners=False) #1/4

        _c12 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c42, _c32, _c22, _c12], dim=1)) #(bt,embed_dim,h,w) 1/4

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w) #(b,t,c,h,w)

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1] #(b,c,h,w)

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        # start_time1=time.time()
        # shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:] #(h,w)
        # c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        # c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        # c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        # c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        # query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1] #list of (b,t-1,c,h,w)


        # query_frame=[query_c1, query_c2, query_c3, query_c4]
        # supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]

        h2=int(h/2)
        w2=int(w/2)
        _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False) #降采样
        _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2)
        
        query_frame=_c_further[:,:-1]
        supp_frame=_c_further[:,-1:]
        # print("query_frame",query_frame.shape,supp_frame.shape)
        # supp_feats,out_cls_mid=self.hypercorre_module(query_frame, supp_frame,img_metas = img_metas) # target: [2 [b,c,h,w]] cluster_centers:[b,t,num_clusters,c,h,w]
        
        out_cls_mid = None

        if self.need_segdeformer:
            supp_feats,out_cls_mid,cluster_centers,mem_out=self.hypercorre_module(query_frame, supp_frame,img_metas = img_metas) # target: [2 [b,c,h,w]] cluster_centers:[b,t,num_clusters,c,h,w]
        else:
            supp_feats,cluster_centers=self.hypercorre_module(query_frame, supp_frame,img_metas = img_metas)

        _c_further2=torch.cat([_c_further[:,-1], supp_feats[0]],1)
        x2 = self.dropout(_c_further2)
        x2 = self.linear_pred1(x2)
        x2=resize(x2, size=(h,w),mode='bilinear',align_corners=False)
        x2=x2.unsqueeze(1)

        if not self.training or not self.aux_loss_decode:
            if self.need_segdeformer:
                x3 = resize(out_cls_mid, size=(h,w),mode='bilinear',align_corners=False).unsqueeze(1)
                # print("x2",x2.shape, "x3",x3.shape,x.shape)
                output=torch.cat([x,x2,x3],dim=1)   ## b*(k+2)*124*h*w
            else:   
                output=torch.cat([x,x2],dim=1)   ## b*(k+2)*124*h*w
        else:
            x4 = [resize(self.linear_pred2(mem_out[i]), size=(h,w),mode='bilinear',align_corners=False).unsqueeze(1) for i in range(len(mem_out))] #[b,t,c,h,w]
            x4 = torch.cat(x4, dim=1) #(b,t,c,h,w)
            if self.need_segdeformer:
                x3 = resize(out_cls_mid, size=(h,w),mode='bilinear',align_corners=False).unsqueeze(1)
                # print("x2",x2.shape, "x3",x3.shape,x.shape)
                output=torch.cat([x,x2,x3,x4],dim=1)   ## b*(k+2)*124*h*w
            else:   
                output=torch.cat([x,x2],dim=1)   ## b*(k+2)*124*h*w

        if not self.training:
            return x2.squeeze(1)
            # return x2.squeeze(1) + 0.5 * x3.squeeze(1)
        return output,cluster_centers
        # return output


class small_decoder2(nn.Module):

    def __init__(self,
                 input_dim=256, hidden_dim=256, num_classes=124,dropout_ratio=0.1):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_classes=num_classes

        self.smalldecoder=nn.Sequential(
            # ConvModule(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            # ConvModule(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(hidden_dim, self.num_classes, kernel_size=1)
            )
        # self.dropout=
        
    def forward(self, input):

        output=self.smalldecoder(input)

        return output
