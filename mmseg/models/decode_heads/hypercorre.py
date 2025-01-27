import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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


def get_sim_mask(query,key,threshold=0.8):
    '''
        input:
            query: [B, C, h, w]
            key: [B, C,t, h, w]
        output:
            sim: [B, h, w, h, w]
            mask: [B, h, w, h, w]
    '''
    b,_,h,w = query.shape
    query = query.flatten(2).transpose(1, 2)  # [B, h*w, C]
    key = key.flatten(2).transpose(1, 2)  # [B, t*h*w, C]
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    sim = query @ key.transpose(1, 2)  # [B, h*w, t*h*w]
    mask_kq = (sim >= threshold).sum(dim=-1) > 0  # [B, h*w], 更新q
    mask_qk = (sim >= threshold).sum(dim=-2) > 0  # [B, t*h*w], 更新k,v
    mask_kq = mask_kq.view(b,h,w) # [B, h, w]
    mask_qk = mask_qk.view(b,-1,h,w) # [B,t, h, w]
    return sim,mask_qk,mask_kq

class Conv3_1_block(nn.Module):
    def __init__(self, in_channels, out_channels,bias=True):
        super(Conv3_1_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1,padding=1,bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, stride=1,bias=bias)
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
        out = self.relu(out)
        out = self.conv2(out)
        return out

class hypercorre_topk2(nn.Module):
    """ top-k2: same selections for each reference image so that attention decoder can be used

    Args:
    num_feats: number of features being used

    """

    def __init__(self,
                 stack_id=None, dim=[64, 128, 320, 512], qkv_bias=True, num_feats=4, backbone='b1',linear_qk = True):
        super().__init__()
        self.stack_id=stack_id
        self.dim=dim
        self.linear_qk=linear_qk
        if linear_qk:
            self.q1 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
            self.q2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
            self.q3 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

            self.k1 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
            self.k2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
            self.k3 = nn.Linear(dim[3], dim[3], bias=qkv_bias)
            self.id = 0
        else:
            self.q1 = Conv3_1_block(dim[1], dim[1])
            self.q2 = Conv3_1_block(dim[2], dim[2])
            self.q3 = Conv3_1_block(dim[3], dim[3])
            self.k1 = Conv3_1_block(dim[1], dim[1])
            self.k2 = Conv3_1_block(dim[2], dim[2])
            self.k3 = Conv3_1_block(dim[3], dim[3])

        self.hpn=HPNLearner_topk2([1,1,1], backbone)
        self.k_top=5
        self.id = 0
        if backbone=='b0':
            self.threh=0.8
        else:
            self.threh=0.5
        
    def forward(self, query_frame, supp_frame,debug=True):
        """ Forward function.
        query_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
        supp_frame: [B*1*c*h/4*w/4, B*1*c*h/8*w/8, B*1*c*h/16*w/16, B*1*c*h/32*w/32]
        Args:
            
        """
        start_time=time.time()
        query_frame=query_frame[::-1]
        supp_frame=supp_frame[::-1]
        query_qkv_all=[]
        query_shape_all=[]

        if self.linear_qk:
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
                query_qkv_all.append(query_qkv.reshape(B,num_ref_clips,hy*wy,cx))       ## B,num_ref_clips,hy*wy,cx
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
                supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx*wx,cx))    ## B,num_ref_clips,hx*wx,cx
                supp_shape_all.append([hx,wx])

        else:
            for ii, query in enumerate(query_frame):
                B,num_ref_clips,cx,hy,wy=query.shape
                query = query.reshape(B*num_ref_clips,cx,hy,wy)
                if ii==0:
                    query_qkv=self.k3(query) # B,num_ref_clips,hy,wy,cx
                elif ii==1:
                    query_qkv=self.k2(query)
                elif ii==2:
                    query_qkv=self.k1(query)
                elif ii==3:
                    ## skip h/4*w/4 feature because it is too big
                    query_qkv_all.append(None)
                    query_shape_all.append([None,None])
                    continue
                query_qkv_all.append(query_qkv.view(B,num_ref_clips,cx,hy*wy).transpose(-1,-2))       ## B,num_ref_clips,hy*wy,cx
                query_shape_all.append([hy, wy])

            supp_qkv_all=[]
            supp_shape_all=[]
            for ii, supp in enumerate(supp_frame):
                B,num_ref_clips,cx,hx,wx=supp.shape
                supp = supp.reshape(B*num_ref_clips,cx,hx,wx)
                if ii==0:
                    supp_qkv=self.q3(supp)    
                elif ii==1:
                    supp_qkv=self.q2(supp)
                elif ii==2:
                    supp_qkv=self.q1(supp)
                elif ii==3:
                    supp_qkv_all.append(None)
                    supp_shape_all.append([None,None])
                    continue
                supp_qkv_all.append(supp_qkv.view(B,num_ref_clips,cx,hx*wx).transpose(-1,-2))    ## B,num_ref_clips,hx*wx,cx
                supp_shape_all.append([hx,wx])

        if debug:
            h,w = query_shape_all[0]
            test_q = supp_qkv_all[0].squeeze(1).transpose(-1,-2).reshape(B,-1,h,w) # [B,c,h,w] 1/32
            test_k = query_qkv_all[0].transpose(-1,-2).reshape(B,3,-1,h,w).permute(0,2,1,3,4) # [B,t,c,n]
            print("shape",test_q.shape,test_k.shape)
            
            sim,mask_qk,mask_kq = get_sim_mask(test_q,test_k)
            p8 = (sim>0.8).sum(-1) > 0
            ratio_32 = p8.sum()/(test_q.shape[-2]*test_q.shape[-1])
            print("ratio_32",ratio_32)
            self.id += 1
            if self.id>10:
                exit()


        loca_selection=[]
        indices_sele=[]

        atten_all=[]
        s_all=[]

        B=supp_qkv_all[0].shape[0]
        q_num_ref=query_qkv_all[0].shape[1]
        for ii in range(0,len(supp_frame)-1):
            hy,wy=query_shape_all[ii]
            hx,wx=supp_shape_all[ii]
            if ii==0:
                atten=torch.matmul(supp_qkv_all[ii], query_qkv_all[ii].transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(hy*wy)
                atten_fullmatrix=atten
            else:
                cx=query_qkv_all[ii].shape[-1]
                query_selected=query_qkv_all[ii]  ## B*(num_clips-1)*(hy*wy)*c
                assert query_selected.shape[:-1]==loca_selection[ii-1].shape     
                
                num_selection=torch.unique((loca_selection[ii-1]).sum(2))
                assert num_selection.shape[0]==1 and num_selection.dim()==1
                query_selected=query_selected[loca_selection[ii-1]>0.5].reshape(B, q_num_ref, int(num_selection[0]),cx)     ##  B*(num_clips-1)*s*c

                atten=torch.matmul(supp_qkv_all[ii], query_selected.transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(s)
            if ii==0:
                atten_topk=torch.topk(atten_fullmatrix,self.k_top,dim=2)[0]    # B*(num_clips-1)*(k)*(hy*wy)
                atten_topk=atten_topk.sum(2)   # B*(num_clips-1)*(hy*wy)
                
                hy_next, wy_next=query_shape_all[ii+1]

                if hy_next==hy and wy_next==wy:
                    s=int(hy*wy*self.threh)
                    indices=torch.topk(atten_topk,s,dim=2)[1]    # B*(num_clips-1)*s
                    topk_mask=torch.zeros_like(atten_topk)
                    topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy*wy)
                    atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)

                else:   # hy_next!=hy or wy_next!=wy
                    atten=atten.reshape(B*q_num_ref,hx*wx,hy,wy)
                    atten=F.interpolate(atten, (hy_next, wy_next), mode='bilinear', align_corners=False).reshape(B,q_num_ref,hx*wx,hy_next*wy_next)

                    atten_topk=atten_topk.reshape(B,q_num_ref,hy,wy)    # B*(num_clips-1)*hy*wy
                    atten_topk=F.interpolate(atten_topk, (hy_next, wy_next), mode='bilinear', align_corners=False)    # B*(num_clips-1)*hy_next*wy_next
                    atten_topk=atten_topk.reshape(B, q_num_ref, hy_next*wy_next)   # B*(num_clips-1)*(hy_next*wy_next)

                    s=int(hy_next*wy_next*self.threh)
                    indices=torch.topk(atten_topk,s,dim=2)[1]    # # B*(num_clips-1)*s
                    topk_mask=torch.zeros_like(atten_topk)
                    topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy_next*wy_next)

                    atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)
                loca_selection.append(topk_mask)
            elif ii<=len(supp_frame)-3:
                loca_selection.append(loca_selection[-1])

            s=atten.shape[3]
            atten=atten.permute(0,1,3,2).reshape(B*q_num_ref*s,hx,wx)   ## (B*(num_clips-1)*s)*hx*wx
            atten_all.append(atten.unsqueeze(1))
            s_all.append(s)
        atten_all=self.hpn(atten_all)
        start_time2=time.time()
        atten_all=atten_all.squeeze(1).reshape(B,q_num_ref,s_all[-1],supp_shape_all[2][0]*supp_shape_all[2][1]).permute(0,1,3,2)   # B*(num_clips-1)*(hx*wx)*s
        return atten_all, loca_selection[-1]>0.5

