import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class CenterPivotConv4d_half(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d_half, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
        #                        bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        ## x should be size of bsz*s, inch, hb, wb

        # if self.stride[2:][-1] > 1:
        #     out1 = self.prune(x)
        # else:
        #     out1 = x
        # bsz, inch, ha, wa, hb, wb = out1.size()
        # out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        # out1 = self.conv1(out1)
        # outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        # out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        # bsz, inch, ha, wa, hb, wb = x.size()
        bsz_s, inch, hb, wb = x.size()
        out2 = self.conv2(x)

        # out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        # out2 = self.conv2(out2)
        # outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        # out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        # if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
        #     out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
        #     out2 = out2.squeeze()

        # y = out1 + out2
        return out2



class HPNLearner_topk2(nn.Module):
    def __init__(self, inch, backbone):
        super(HPNLearner_topk2, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d_half(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # ## new way for better trade-off between speed and performance
        if backbone=='b1':
            outch1, outch2, outch_final = 1,2,1
            self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2], [3, 3], [1, 1])
            self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2], [3, 3], [1, 1])
            self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2], [5, 3], [1, 1])

            # Mixing building blocks
            self.encoder_layer4to3 = make_building_block(outch2, [outch2, outch2], [3, 3], [1, 1])
            self.encoder_layer3to2 = make_building_block(outch2, [outch2, outch_final], [3, 3], [1, 1])
        else:
            outch1 = 1
            self.encoder_layer4 = make_building_block(inch[0], [outch1], [3], [1])
            self.encoder_layer3 = make_building_block(inch[1], [outch1], [5], [1])
            self.encoder_layer2 = make_building_block(inch[2], [outch1], [5], [1])

            # # Mixing building blocks
            self.encoder_layer4to3 = make_building_block(outch1, [1], [3], [1])
            self.encoder_layer3to2 = make_building_block(outch1, [1], [3], [1])


    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz_s, ch,  hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        # o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr


    def forward(self, hypercorr_pyramid):
        ## atten shape: bsz_s,inch,hx,wx

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        # hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        # hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        return hypercorr_mix432

def get_sim_mask(query,key,threshold=0.95):
    '''
        input:
            query: [B, 1,C, h, w]
            key: [B, t,C, h, w]
        output:
            sim: [B, t, h, w, h, w]
            mask: [B, t, h, w, h, w]
    '''
    b,_,_,h,w = query.shape
    query = query.flatten(3).transpose(-1, -2)  # [B,1, h*w, C]
    key = key.flatten(3).transpose(-1, -2)  # [B, t,h*w, C]
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    sim = query @ key.transpose(-1, -2)  # [B,t, h*w, h*w]
    mask_kq = (sim >= threshold).sum(dim=-1) > 0  # [B, t, h*w], 更新q
    mask_qk = (sim >= threshold).sum(dim=-2) > 0  # [B, t, h*w], 更新k,v
    mask_kq = mask_kq.view(b,-1,h,w) # [B,t, h, w]
    mask_qk = mask_qk.view(b,-1,h,w) # [B,t, h, w]
    return sim,mask_qk,mask_kq


def StaticDynamicAttention(query,key,value, lv=[1,1], low_res_weights=None, H1=None,W1=None):
    '''
        # 统计结果：
        # 分辨率越大的情况下相似度越高,mask的区域越多,对于高分辨率进行局部注意力有效,低分辨率进行全局注意力有效
        # vspw数据集在一个视频中在1/4块下,有80%的相似度的概率为0.83,随着跨帧距离越远,概率越小
        # 距离越近，高度冗余，大部分是非剧烈运动(静态情况)。块状单独运算可以有效聚合局部静态的信息
        # 不剧烈的分块计算,测试：切块
        # 剧烈运用大面积权重
        # 重新排列高分辨率特征图，使其按块处理
        lv:控制切块大小,切块数目[num_h,num_w],[1,1]表示使用全注意力
        query: [B, 1,C, H, W]
        key: [B,  T,C, H, W]
        value: [B,T,C, H, W]
    '''
  
    # B, N, TN = low_res_weights.shape
    B,  T, C, H, W = value.shape
    h_ration = lv[0]  
    w_ration = lv[1]
    # 检查N1是否是N的4倍
    # assert T*H*W == h_ration * w_ration * TN, "高分辨率特征图的分辨率必须是低分辨率的倍数。"
    # 分t帧进行局部更新，同时获取t帧的mask
    # out = torch.zeros_like(query).to(query.device)
    out = torch.zeros(B,T,C,H,W).to(query.device)
    none_update_query = []
    none_update_value = []
    none_update_key = []
    none_point_indices = []
    size_h = H // h_ration
    size_w = W // w_ration
    ratios_out = []
    # print("lv",lv)
    # print(H,W,size_h,size_w)
    for j in range(h_ration): # 切块
        for i in range(w_ration):
            start_h,end_h,start_w,end_w = j*size_h,(j+1)*size_h,i*size_w,(i+1)*size_w
            if j == h_ration - 1 or end_h > H:
                end_h = H
            if i == w_ration - 1 or end_w > W:
                end_w = W
                
            if start_h >= end_h or start_w >= end_w: # 没有了
                continue
            # print("start_h,end_h,start_w,end_w",start_h,end_h,start_w,end_w)
          # 直接把t展开算的
            query_block = query[:, :, :, start_h:end_h, start_w:end_w].expand(-1,T,-1,-1,-1) # [B,1,C,h,w]
            key_block = key[:, :, :, start_h:end_h, start_w:end_w] # [B,t,C,h,w]
            value_block = value[:, :, :, start_h:end_h, start_w:end_w] # [B,t,C,h,w]
            # 计算相似度
            # print("getting sim:",query_block.shape,key_block.shape)
            sim,mask_qk,mask_kq = get_sim_mask(query_block,key_block) 
            # sim:[B,t,h*w,h*w], mask_kq:[B,t,h,w],mask_qk:[B,t,h,w],qk是哪些k对整个q有用更新k,v,kq更新q,mask_kq表示前面的每一帧对当前帧哪些位置有用，而mask_qk表示当前帧对前面的每一帧哪些位置有用
          
            # 用更新的k，v计算这个区域
            # 这里可以有两种：
            # 1. 利用k,v的mask更新k,v 
            attn_block = sim.softmax(dim=-1) # [B,t,h*w,h*w]
            # print("attn",attn_block.shape)
            out_block = attn_block @ value_block.flatten(3).transpose(-1, -2) # [B,t,h*w,hw] * [B,t,hw,C] = [B,t,h*w,C]
            # print("out_block",out_block.shape)  
            # print("end-start",end_h-start_h,end_w-start_w)
            out_block = out_block.transpose(-1, -2).view(B,T,C,end_h-start_h,end_w-start_w) #这是前面每一帧对当前帧的更新结果
            out[:, :, :,start_h:end_h, start_w:end_w] = out_block 
            
            # 2.直接用sim结果，避免重复计算， 看冗余程度测试结果
            # value_block = value_block[mask_qk.unsqueeze(1).expand(-1,C,-1,-1)].view(B,C,-1) # [B,C,s]
            # print(value_block.shape)
            # key_block = key_block[mask_qk.unsqueeze(1).expand(-1,C,-1,-1)].view(B,C,-1) # [B,C,s]
            # query_blocck = query_block.flatten(2).transpose(1, 2) # [B,h*w,,C]
            # attn_block = query_block @ key_block # [B,h*w,S]
            # attn_block = F.softmax(attn_block, dim=-1)
            # out_block = attn_block @ value_block.transpose(1, 2) # [B,h*w,C]

            # 更新比较差的区域：mask_kq,需要全局信息，需要将全部的t展开得到结果
            # 从query_block中取出mask_kq为False的区域以及对应的值(还是结果的out_block???)
            # print("relation_ratio:",mask_kq.sum()/mask_kq.numel())
            ratios_out.append(mask_kq.flatten(2).sum(-1)/((end_h-start_h)*(end_w-start_w))) #[B,T]
            none_mask_kq = ~mask_kq #[b,t,H,W]
            # print("mask",none_mask_kq.shape,query_block.shape)
            # 使用掩码选择数据
            query_block_data = [[query_block[b, t,:, none_mask_kq[b,t]] for t in range(T)] for b in range(B)]#[b [t (C,S')]]
            # print("query_block_data",query_block_data[0].shape,len(query_block_data))
            none_update_query.append(query_block_data) #[B,C,S]
            # print("len none_update_query",len(none_update_query))
            # 位置获取更新的
            none_point_indices.append(none_mask_kq)
            
            # k,v,包含t帧的全部数据
            none_mask_qk = ~mask_qk #[b,t,h,w]
            # print("mask",none_mask_qk.shape)
            key_block_data = [[key_block[b, t,:, none_mask_qk[b,t]] for t in range(T)] for b in range(B)] #[b [t (C,S)]]
            # print("key_block_data",key_block_data[0].shape,len(key_block_data))
            none_update_key.append(key_block_data) 
            value_block_data = [[value_block[b, t,:, none_mask_qk[b,t]] for t in range(T)] for b in range(B)] #[b [t (C,S)]]
            none_update_value.append(value_block_data) 
            # query_block_data = torch.stack([query_block[b, :, none_mask_kq[b]] for b in range(B)], dim=0) # [B,C,S1]
            # none_update_key.append(key_block[none_mask_qk.unsqueeze(1).expand(-1,C,-1,-1)]) # [B,C,S1]
            # none_update_value.append(value_block[none_mask_qk.unsqueeze(1).expand(-1,C,-1,-1)]) #[B,C,S1]
    # print("out1",out)
    # 获取完所有区域的更新值
    # print("ratios_out",ratios_out)
    for ii in range(B):
        for t in range(T):
            query_out = []
            key_out = []
            value_out = []
            mask_out = torch.zeros((H,W),dtype=torch.bool).to(query.device)
            x,y=0,0
            for query_block_data,key_block_data,value_block_data,mask_data in zip(none_update_query,none_update_key,none_update_value,none_point_indices):
                query_out.append(query_block_data[ii][t]) #[(C,S) ] 
                key_out.append(key_block_data[ii][t]) #[(C,S1)]
                value_out.append(value_block_data[ii][t]) #[(C,S1)]
                h,w = mask_data[ii][t].shape
                mask_out[x:x+h,y:y+w] = mask_data[ii][t] #[H,W]
                y = (y + w) % W
                x = (x + h) % H if y == 0 else x
            assert x == 0 and y == 0, "更新的位置和更新的值不匹配"
            query_out = torch.cat(query_out,dim=-1) # [C,S_all]
            key_out = torch.cat(key_out,dim=-1)  # [C,S1_all]
            value_out = torch.cat(value_out,dim=-1) # [C,S1_all]
            # print("shape",query_out.shape,key_out.shape,value_out.shape,mask_out.shape)
            atten = query_out.transpose(0,1) @ key_out # [S_all,S1_all]
            # atten = atten * (C ** -0.5)
            atten = F.softmax(atten,dim=-1)
            _out = atten @ value_out.transpose(0,1).contiguous() # [S_all,C]
            _out = _out.transpose(0,1).contiguous().view(C,-1) #[C,S_all]
            # print("shape_out",_out.shape)
            # print(mask_out)
            # print(_out)
            assert mask_out.sum() == _out.shape[-1], "更新的位置和更新的值不匹配"
            expanded_mask = mask_out.unsqueeze(0)  # (1, H, W)
            # 使用 masked_scatter_ 直接更新所有通道的数据
            out[ii,t].masked_scatter_(expanded_mask, _out)
            # out[ii].masked_scatter_(mask_out.flatten(),_out)
            # print("out",out[ii])
    return out

class Conv3_1_block(nn.Module): # 既然linear和point_wise类似，那么采用可分离卷积处理
    def __init__(self, in_channels, out_channels,bias=True):
        super(Conv3_1_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1,bias=False,padding=1,groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, stride=1,bias=bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)
        out = self.conv2(out)
        return out

# cnn-qk,eatch t, 分开算的时间
class hypercorre_topk2(nn.Module):
    """ top-k2: same selections for each reference image so that attention decoder can be used

    Args:
    num_feats: number of features being used

    """

    def __init__(self,
                 stack_id=None, dim=[64, 128, 320, 512], qkv_bias=True, num_feats=4, backbone='b1',linear_v = True,embedding_dim=None):
        super().__init__()
        self.stack_id=stack_id
        self.dim=dim
        # self.q1 = nn.Linear(dim[1], dim[1], bias=qkv_bias) #这里过linear之后数值好像改变了很多,这里的全局是有利的吗？，换成CNN层可以更好的保留局部信息
        # self.q2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        # self.q3 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        # self.k1 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        # self.k2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        # self.k3 = nn.Linear(dim[3], dim[3], bias=qkv_bias)
        self.q1 = Conv3_1_block(dim[1], dim[1])
        self.q2 = Conv3_1_block(dim[2], dim[2])
        self.q3 = Conv3_1_block(dim[3], dim[3])
        self.k1 = Conv3_1_block(dim[1], dim[1])
        self.k2 = Conv3_1_block(dim[2], dim[2])
        self.k3 = Conv3_1_block(dim[3], dim[3])
        if embedding_dim is None:
            embed_dim = [dim[1],dim[2],dim[3]]
            self.embed_dim = embed_dim
        else:
            embed_dim = [embedding_dim,embedding_dim,embedding_dim]
            self.embed_dim = embed_dim
        if not linear_v:
            self.v1 = Conv3_1_block(dim[1], embed_dim[0])
            self.v2 = Conv3_1_block(dim[2], embed_dim[1])
            self.v3 = Conv3_1_block(dim[3], embed_dim[2])
            
        self.id = 0
        self.linear_v = linear_v
        if backbone=='b0':
            self.threh=0.8
        else:
            self.threh=0.5
        
    def forward(self, query_frame, supp_frame,v_frames,lv = [[1,1],[2,2],[4,4],[8,8]],debug = False):
        """ Forward function.
        query_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
        supp_frame: [B*1*c*h/4*w/4, B*1*c*h/8*w/8, B*1*c*h/16*w/16, B*1*c*h/32*w/32]
        v_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
        Args:
            
        """
        start_time=time.time()
        query_frame=query_frame[::-1] #[B*(num_clips-1)*c*h/32*w/32, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/4*w/4]
        supp_frame=supp_frame[::-1]
        v_frames = v_frames[::-1]
        if debug:
            test_q = supp_frame[0].squeeze(1) # [B,c,h,w] 1/32
            test_k = query_frame[0].permute(0,2,1,3,4).squeeze(1) # [B,c,t,h,w]
            print("shape",test_q.shape,test_k.shape)
            
            sim,mask_qk,mask_kq = get_sim_mask(test_q,test_k)
            p8 = (sim>0.8).sum(-1) > 0
            ratio_32 = p8.sum()/(test_q.shape[-2]*test_q.shape[-1])
            print("ratio_32",ratio_32)

            test_q = supp_frame[1].squeeze(1) # [B,c,h,w]
            test_k = query_frame[1].permute(0,2,1,3,4).squeeze(1) # [B,c,t,h,w]
            print("shape2",test_q.shape,test_k.shape)

            sim,mask_qk,mask_kq = get_sim_mask(test_q[:,:,:test_q.shape[-2]//2,:test_q.shape[-1]//2],test_k[:,:,:,:test_k.shape[-2]//2,:test_k.shape[-1]//2])
            p8 = (sim>0.8).sum(-1) > 0
            ratio_16_lu = p8.sum()/(test_q.shape[-2]//2*test_q.shape[-1]//2)
            print("ratio_16_lu",ratio_16_lu)

            test_q = supp_frame[2].squeeze(1) # [B,c,h,w]
            test_k = query_frame[2].permute(0,2,1,3,4).squeeze(1) # [B,c,t,h,w]
            print("shape3",test_q.shape,test_k.shape)

            sim,mask_qk,mask_kq = get_sim_mask(test_q[:,:,:test_q.shape[-2]//4,:test_q.shape[-1]//4],test_k[:,:,:,:test_k.shape[-2]//4,:test_k.shape[-1]//4])
            p8 = (sim>0.8).sum(-1) > 0
            ratio_8_lu = p8.sum()/(test_q.shape[-2]//4*test_q.shape[-1]//4)
            print("ratio_8_lu",ratio_8_lu)
            self.id = self.id+1
            if self.id>10:
                exit()

        query_qkv_all=[]
        query_shape_all=[]
        v_frame_qkv_all=[]

        '''
            ori mrcfa:
        for ii, query in enumerate(query_frame):
            B,num_ref_clips,cx,hy,wy=query.shape
            if ii==0:
                query_qkv=self.k3(query.permute(0,1,3,4,2))
            elif ii==1:
                query_qkv=self.k2(query.permute(0,1,3,4,2))
            elif ii==2:
                query_qkv=self.k1(query.permute(0,1,3,4,2))
            elif ii==3:
                ## skip h/4*w/4 feature because it is too big
                query_qkv_all.append(None)
                query_shape_all.append([None,None])
                continue
            query_qkv_all.append(query_qkv.reshape(B,num_ref_clips,hy,wy,cx))       ## B,num_ref_clips,hy,wy,cx
            query_shape_all.append([hy, wy])

        supp_qkv_all=[]
        supp_shape_all=[]
        for ii, supp in enumerate(supp_frame):
            B,num_ref_clips,cx,hx,wx=supp.shape
            if ii==0:
                supp_qkv=self.q3(supp.permute(0,1,3,4,2))    
            elif ii==1:
                supp_qkv=self.q2(supp.permute(0,1,3,4,2))
            elif ii==2:
                supp_qkv=self.q1(supp.permute(0,1,3,4,2))
            elif ii==3:
                supp_qkv_all.append(None)
                supp_shape_all.append([None,None])
                continue
            supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx,wx,cx))    ## B,1,hx,wx,cx
            supp_shape_all.append([hx,wx])
        '''

        for ii, (query,v_frame) in enumerate(zip(query_frame,v_frames)):
            B,num_ref_clips,cx,hy,wy=query.shape
            v_channels = self.embed_dim[2-ii]
            if ii==0:
                query_qkv=self.k3(query.reshape(B*num_ref_clips,cx,hy,wy))
                if not self.linear_v:
                    v_frame_qkv = self.v3(v_frame.reshape(B*num_ref_clips,cx,hy,wy))
            elif ii==1:
                query_qkv=self.k2(query.reshape(B*num_ref_clips,cx,hy,wy))
                if not self.linear_v:
                    v_frame_qkv = self.v2(v_frame.reshape(B*num_ref_clips,cx,hy,wy))
            elif ii==2:
                query_qkv=self.k1(query.reshape(B*num_ref_clips,cx,hy,wy))
                if not self.linear_v:
                    v_frame_qkv = self.v1(v_frame.reshape(B*num_ref_clips,cx,hy,wy))
            elif ii==3:
                ## skip h/4*w/4 feature because it is too big
                query_qkv_all.append(None)
                if not self.linear_v:
                    v_frame_qkv_all.append(None)
                query_shape_all.append([None,None])
                continue
            query_qkv_all.append(query_qkv.reshape(B,num_ref_clips,cx,hy,wy))       ## B,num_ref_clips,cx,hy,wy
            if not self.linear_v:
                v_frame_qkv_all.append(v_frame_qkv.reshape(B,num_ref_clips,v_channels,hy,wy))
            query_shape_all.append([hy, wy])

        supp_qkv_all=[]
        supp_shape_all=[]
        for ii, supp in enumerate(supp_frame):
            B,num_ref_clips,cx,hx,wx=supp.shape
            if ii==0:
                supp_qkv=self.q3(supp.reshape(B*num_ref_clips,cx,hx,wx))    
            elif ii==1:
                supp_qkv=self.q2(supp.reshape(B*num_ref_clips,cx,hx,wx))
            elif ii==2:
                supp_qkv=self.q1(supp.reshape(B*num_ref_clips,cx,hx,wx))
            elif ii==3:
                supp_qkv_all.append(None)
                supp_shape_all.append([None,None])
                continue
            supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,cx,hx,wx))    ## B,1,cx,hx,wx
            supp_shape_all.append([hx,wx])

        out = []
        B=supp_qkv_all[0].shape[0]
        # q_num_ref=query_qkv_all[0].shape[1]
        # 空间不具有等价性
        for ii in range(0,len(supp_frame)-1): #1/32 - 1/8 切块lv[ii]
            hy,wy=query_shape_all[ii] 
            hx,wx=supp_shape_all[ii]
            query = supp_qkv_all[ii]  #[b,1,c,h,w]
            key = query_qkv_all[ii]  #[b,num_ref,c,h,w]
            if not self.linear_v:
                value = v_frame_qkv_all[ii]
            else:
                value = v_frames[ii] #[b,num_ref,c,h,w]
            B,num_ref_clips,cx,hx,wx=value.shape
            # print("shape",query.shape,key.shape,value.shape)
            # 时间不具有等价性
            dy_out = StaticDynamicAttention(query=query,key=key,value=value,lv=lv[ii]) #[b,t,c,h,w]
            out.append(dy_out.reshape(-1,cx,hx,wx)) #[[b*t,c,h/32,w/32],[b*t,c,h/16,w/16],[b*t,c,h/8,w/8]]
        # 构建时序关系结果或者是空间关系结果？
        return out
