import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .transformer_module import SelfAttentionLayer, CrossAttentionLayer, FFNLayer
from mmseg.models.utils import SelfAttentionBlockWithTime
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from .fdsf import FDSF,PagFM

#参照CAT的实现，有效的进行块内计算注意力的同时，计算了块间注意力的时候需要处理，块内部linear实现，可以保持一定的全局
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


def partition(x, patch_size): # 已检查
    """
    Args:
        x: (B, H, W, C)
        patch_size (int): patch size

    Returns:
        patches: (num_patches*B, patch_size, patch_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    patches = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, patch_size, patch_size, C)
    return patches


def reverse(patches, patch_size, H, W): #已检查
    """
    Args:
        patches: (num_patches*B, patch_size, patch_size, C)
        patch_size (int): Patch size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(patches.shape[0] / (H * W / patch_size / patch_size))
    x = patches.view(B, H // patch_size, W // patch_size, patch_size, patch_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,need_dw=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.need_dw = need_dw
        if need_dw:
            self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.need_dw:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchProjection(nn.Module):
    """ Patch Projection Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

class Attention(nn.Module):
    """ Basic attention of IPSA and CPSA.

    Args:
        dim (int): Number of input channels.
        patch_size (tuple[int]): Patch size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        attn_drop (float, optional): Dropout ratio of attention weight.
        proj_drop (float, optional): Dropout ratio of output.
        rpe (bool): Use relative position encoding or not.
    """

    def __init__(self, dim, patch_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size  # Ph, Pw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rpe = rpe

        if self.rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * patch_size[0] - 1) * (2 * patch_size[1] - 1), num_heads))  # 2*Ph-1 * 2*Pw-1, nH

            # get pair-wise relative position index for each token inside one patch
            coords_h = torch.arange(self.patch_size[0])
            coords_w = torch.arange(self.patch_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Ph, Pw
            coords_flatten = torch.flatten(coords, 1)  # 2, Ph*Pw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Ph*Pw, Ph*Pw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ph*Pw, Ph*Pw, 2
            relative_coords[:, :, 0] += self.patch_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.patch_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.patch_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Ph*Pw, Ph*Pw
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x,memory=None):
        """
        Args: input_cpsa torch.Size([3072, 1, 225]) torch.Size([3072, 1, 225])
            x: input features with shape of (num_patches*B, N, C)
            memory: input features with shape of (num_patches*B*T, N, C)
        """
        B_, N, C = x.shape
        memory = memory.view(B_, -1, N, C)  # B, T,N, C # 一起起效果，还是加上局部的time_refine? for time_refine
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).unsqueeze(1) # nP*B, nH, N, C
        kv = self.kv(memory).reshape(B_, -1, N, 2, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5) # nP*B, T, nH, N, C
        k, v = kv[0], kv[1]
        q = q * self.scale
        # print("q,k,v",q.shape,k.shape,v.shape)
        attn = (q @ k.transpose(-2, -1)) #(nP*B, T, nH, N, N)

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.patch_size[0] * self.patch_size[1], self.patch_size[0] * self.patch_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0).unsqueeze(0)  # nP*B, T, nH, N, N 

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B_,-1, N, C) # [nP*B, T,nH,  N, N] * [nP*B, T, nH,N, C] -> [nP*B, T,nH, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x #[nP*B, T, N, C]

# depth-wise conv
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# 对原型求交叉注意力，过分依赖于原型的提取效果，且是两阶段的训练，动态的全局注意力获取更加方便简洁，CFFM++结构的先进之处在于不同帧之间语义的对齐，
# 原型中包含的信息也足够覆盖全部类别，对于新出现的类别会更加友好
# 如果把CFFM的语义对齐加入，全局分割也修改为当前cpsa代码，动态的全局性质。
# CFFM中计算交叉注意力的时候加上了自身

class CATBlock(nn.Module):
    """ Implementation of CAT Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        patch_size (int): Patch size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        drop (float, optional): Dropout rate.
        attn_drop (float, optional): Attention dropout rate.
        drop_path (float, optional): Stochastic depth rate.
        act_layer (nn.Module, optional): Activation layer.
        norm_layer (nn.Module, optional): Normalization layer.
        rpe (bool): Use relative position encoding or not.
    """

    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type="ipsa", rpe=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.attn_type = attn_type

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim=dim if attn_type == "ipsa" else self.patch_size ** 2, patch_size=to_2tuple(self.patch_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpe=rpe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # need to be changed in different stage during forward phase
        self.H = None
        self.W = None
        
    def forward(self, x,memory = None):
        '''
            x:[b,n,c]
            memory:[b*t,n,c]
        '''
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        if memory is None:
            memory = x

        _B,L,C = memory.shape #b*t,h*w,c
        shortcut = x
        x = self.norm1(x)
        memory = self.norm1(memory) 
        x = x.view(B, H, W, C)
        memory = memory.view(_B, H, W, C)
        # padding to multiple of patch size in each layer
        pad_l = pad_t = 0
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        memory = F.pad(memory, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        # print("pad input",x.shape,memory.shape)
        # partition
        patches = partition(x, self.patch_size)  # Bt*nP, patch_size, patch_size, C
        patches = patches.view(-1, self.patch_size * self.patch_size, C)  # Bt*nP, patch_size*patch_size, C
        patches_memory = partition(memory, self.patch_size)  # B*TnP, patch_size, patch_size, C
        patches_memory = patches_memory.view(B,-1,Hp//self.patch_size*Wp//self.patch_size, self.patch_size, self.patch_size, C)  # nP*B*T, patch_size*patch_size, C
        patches_memory = patches_memory.permute(0,2,1,3,4,5).contiguous().view(-1, self.patch_size * self.patch_size, C)  # nP*B*T, patch_size*patch_size, C
        # print("partition",patches.shape,patches_memory.shape)
        # IPSA or CPSA
        if self.attn_type == "ipsa":
            attn = self.attn(patches,patches_memory)  # [B*nP, T, N, C] 
            attn = attn.view(B, Hp//self.patch_size*Wp//self.patch_size,-1, self.patch_size * self.patch_size, C)  # [B,np, T,n,C]
            attn = attn.permute(0,2,1,3,4).contiguous().view(-1, self.patch_size * self.patch_size, C)  # [B*T, nP, patch_size*patch_size, C]
        
        # 第一个ipsa之后进行cpsa,进行全局补充，之后加入时间信息进行交互
        elif self.attn_type == "cpsa":
            patches = patches.view(B, (Hp // self.patch_size) * (Wp // self.patch_size), self.patch_size ** 2, C).permute(0, 3, 1, 2).contiguous()
            patches = patches.view(-1, (Hp // self.patch_size) * (Wp // self.patch_size), self.patch_size ** 2) # B*C, nP*nP, patch_size*patch_size
            patches_memory = patches_memory.view(B,(Hp // self.patch_size) * (Wp // self.patch_size),-1,self.patch_size ** 2, C).permute(0,4,2,1,3).contiguous()
            patches_memory = patches_memory.view(-1, (Hp // self.patch_size) * (Wp // self.patch_size), self.patch_size ** 2) # B*C*T, nP*nP, patch_size*patch_size
            #[nP*B, T, N, C]
            # print("input_cpsa",patches.shape,patches_memory.shape) #input_cpsa torch.Size([3072, 1, 225]) torch.Size([3072, 1, 225])
            attn = self.attn(patches,patches_memory).view(B, C,-1, (Hp // self.patch_size) * (Wp // self.patch_size), self.patch_size ** 2)
            attn = attn.permute(0,2,3,4,1).contiguous().view(-1, self.patch_size ** 2, C) # B*T*nP, patch_size*patch_size, C
            
        else :
            raise NotImplementedError(f"Unkown Attention type: {self.attn_type}")

        # print("attn_out",attn.shape)
        # reverse opration of partition
        attn = attn.view(-1, self.patch_size, self.patch_size, C)
        x = reverse(attn, self.patch_size, Hp, Wp)  # B*T H'*W' C
        
        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(-1, H * W, C)
        assert x.shape[0] == _B, "output feature has wrong size"
        
        # FFN
        x = x.view(B, -1, H*W, C)
        x = shortcut.unsqueeze(1) + self.drop_path(x) #[B, T, N, C]
        x = x.flatten(0, 1)  # [B*T, N, C]
        # print("outx",x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x),H,W))
        return x.view(-1,H*W, C) #[B*T, N, C]


# 第一层计算ipsa,分时间分块计算，第二层先计算spsa,补充对于每个t的全局信息，再加上时间信息进行交互，获取最后多尺度结果，最后将多尺度结果进行融合
# layer1(ipsa): b*t,n,c
# layer2(cpsa): b*t,n,c
# layer3(time_refine): b*t,n,c

class TimeLayer(nn.Module): #CNN:short_time, sa:long_time / (long_time加权)
    def __init__(self,hidden_channel=256,feedforward_channel=2048,num_head=8,decoder_layer=6,act_layer=nn.GELU,need_pre_memory=True):
        super(TimeLayer, self).__init__()
        self.transformer_time_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.conv_short_aggregate_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        self.num_head = num_head
        self.num_layers = decoder_layer
        self.need_pre_memory = need_pre_memory
        for _ in range(decoder_layer):
            self.transformer_time_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False, # True from poolformer
                 )
            )

            self.conv_short_aggregate_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channel, hidden_channel,
                              kernel_size=5, stride=1,
                              padding='same', padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel,
                              kernel_size=3, stride=1,
                              padding='same', padding_mode='replicate'),
                )
            )

            self.conv_norms.append(nn.LayerNorm(hidden_channel))
            if need_pre_memory:
                self.transformer_cross_attention_layers.append( #和backbone的输出进行语义对其
                    CrossAttentionLayer(
                        d_model=hidden_channel,
                        nhead=num_head,
                        dropout=0.0,
                        normalize_before=False,
                    )
                )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_channel)


    def forward(self,x,memory,t):
        '''
            x: [B*T,N,C]
            memory: [B*T,N,C] ,feature maps from backbone
        '''
        outputs = []
        # print('in_time_layer',x.shape,memory.shape,t)
        memory = memory.view(-1,t,memory.shape[-2],memory.shape[-1]).flatten(1,2) #[B,T*N,C]
        output = x

        for i in range(self.num_layers):
            output = output.view(-1,t,x.shape[-2],x.shape[-1]) #[B,T,N,C]
            # print("x.shape",output.shape)
            # output = x.permute(1, 0, 2, 3).flatten(1, 2)  # (t, b*q, c) #为什么长程是 b*q,c 为不是 t*q,c
            output = output.flatten(1, 2)  #(b, tn, c)
            # do long temporal attention
            output = self.transformer_time_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            # do short temporal conv 
            output = output.view(-1, t, x.shape[-2], x.shape[-1]).permute(0,2,1,3)  # (b, t, n, c) -> (b, n, t, c)
            output = output.flatten(0, 1).transpose(-1, -2)  # (bn, c, t)
            output = self.conv_norms[i](
                (self.conv_short_aggregate_layers[i](output) + output).transpose(1, 2)
            ).transpose(1, 2) #(bn,c,t)

            output = output.transpose(-1,-2).reshape(-1,t,x.shape[-2],x.shape[-1]).flatten(1,2) #(b, t*n, c)
            
            # do cross attention with ori feature enhuns
            if self.need_pre_memory:
                output = self.transformer_cross_attention_layers[i](
                    output, memory,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )  #(b, t*n, c)

            output = output.reshape(-1, x.shape[-2], x.shape[-1])  # (b*t, n, c)
            outputs.append(output)

        return outputs[-1] # (b*t, n, c)

class TemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        '''
            x: [B, TC,H,W]
        '''
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x) # avg * conv_spatial
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x
    
class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        f_x = self.conv1(attn)         # 1x1 conv
        # append a se operation
        b, c, _, _ = x.size()
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
        avg_pool = self.fc(avg_pool).view(b, c, 1, 1)
        max_pool = self.fc(max_pool).view(b, c, 1, 1)
        attn = self.sigmoid(avg_pool + max_pool)
        return attn * f_x * u

class TimeLayer2(nn.Module): #CNN:short_time, sa:long_time / (long_time加权) / deformable attention 采样
    def __init__(self,dim,t):
        super(TimeLayer2, self).__init__()
        self.dim = dim
        self.time_attn = TemporalAttention(dim*t,kernel_size=13)

    def forward(self,x,t,H,W):
        '''
        x: [B*T,N,C]
        '''
        _B,_,C = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(-1,t*x.shape[-2],H,W) #[B,TC,H,W]
        out = self.time_attn(x) #[B,TC,H,W]
        return out.view(_B,C,-1).transpose(1, 2) #[B*T,N,C]


# 原型一致化时间,from A Transformer-based Decoder for Semantic Segmentation with Multi-level Context Mining iccv22
# 设计能够有效缓解segformer的尺度拼接问题
class Class_Token_Seg3(nn.Module):
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, num_classes=150, qkv_bias=True, qk_scale=None, T=3):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.zeros(1, T, num_classes, dim)) #这里需要T来表示不同帧里面的类别可变性，也像类别原型一样
        
        # test 这里的原型应该产生于类别，还是类别产生于原型，还是互相不影响
        self.prop_token = nn.Parameter(torch.zeros(1, num_classes+26, dim)) #原型，需要扩张到t的表示原型的一致性串通时间维度,这里可以适当增大原型数目进行测试
        self.T = T
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.prop_token, std=.02)


    def forward(self, x):#, x1):
        b, t, c, h, w = x.size()
        assert t == self.T, "Input tensor has wrong time"
        x = x.flatten(3).transpose(-1, -2) #[b,t,n,c]
        B, T, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1, -1) # [B, T, num_classes, dim]
        prop_tokens = self.prop_token.unsqueeze(1).expand(B, T, -1, -1) # [B, T, num_classes, dim]
        
        x = torch.cat((cls_tokens, x), dim=2) #[B, T, num_classes + N, dim]
        B, T, N, C = x.shape
        q = self.q(x).reshape(B, T, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = self.k(x[:, :, 0:self.num_classes]).unsqueeze(1).reshape(B, T, self.num_classes, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4) #[B, T, num_heads, num_classes, dim]
        
        k = k * self.scale
        attn = (k @ q.transpose(-2, -1)).squeeze(2).transpose(-2, -1) # [B, T, num_classes + N, num_classes]
        attn = attn[:,:, self.num_classes:] #[B, T, N, num_classes]
        x_cls = attn.permute(0, 1, 3, 2).reshape(b, t, -1, h, w)
        return x_cls, prop_tokens



class TransformerClassToken3(nn.Module):

    def __init__(self, dim, num_heads=4, num_classes=150, depth=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_cfg=None, norm_cfg=None, sr_ratio=1, trans_with_mlp=True, att_type="SelfAttention"):
        super().__init__()
        self.trans_with_mlp = trans_with_mlp
        self.depth = depth
        print("TransformerOriginal initial num_heads:{}; depth:{}, self.trans_with_mlp:{}".format(num_heads, depth, self.trans_with_mlp))   
        self.num_classes = num_classes

        # test 原型数目
        self.num_prototypes = 26+num_classes

        self.linear_cls = nn.Linear(num_classes,self.num_prototypes)

        self.cross_attn = SelfAttentionBlockWithTime(
            key_in_channels=dim,
            query_in_channels=dim,
            channels=dim,
            out_channels=dim,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=None,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'))
        
        #self.conv = nn.Conv2d(dim*3, dim, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(dim*2,dim, kernel_size=3, stride=1, padding=1)
        self.apply(self._init_weights) 
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
             
    def forward(self, x, cls_tokens, out_cls_mid):
        b, t, c, h, w = x.size()
        out_cls_mid = out_cls_mid.flatten(3).transpose(-1, -2)   # [b,t,n,c]
        # #within images attention
        # x1 = self.attn(x, x) # 前面做了
        #cross images attention
        out_cls_mid = self.linear_cls(out_cls_mid) #b,t,n,num_classes -> b,t,n,num_prototypes
        out_cls_mid = out_cls_mid.softmax(dim=-1)
        # print("out_cls_mid",out_cls_mid.shape)
        # print(self.linear_cls)
        # test num of prototypes

        cls = out_cls_mid @ cls_tokens #b,t,n,num_prototypes @ b,t,num_prototypes,C -> b,t,n,C c:原型数目
        
        cls = cls.permute(0, 1, 3, 2).reshape(b, t, c, h, w) # 原型表征,用同一组原型串通时间
        x2 = self.cross_attn(x, cls)

        x = x+x2
        
        return x


# CATBlock： ipsa: 建模静态块内， cpsa:建模全局的动态行为(这里应该有交互吗？） time_layer:建模时间交互， out：linear融合多个时间结果，需要考虑空间？

# 多尺度的交互： 逐渐上采样的过程中，可以有avg(x) - x来增强细节和边界(以及证明)
class hypercorre_topk2(nn.Module):
    """ top-k2: same selections for each reference image so that attention decoder can be used

    Args:
    num_feats: number of features being used

    """

    def __init__(self,dim=[64, 128, 320, 512], num_layers=1, t=3, time_decoder_layer=3,embedding_dim=256,num_classes = 124,ratio_fusio = False):
        super().__init__()
        self.dim=dim
        self.pre_isa_blocks = nn.ModuleList()
        self.cpa_blocks = nn.ModuleList()
        self.post_isa_blocks = nn.ModuleList()
        self.tmp_blocks = nn.ModuleList()
        self.conv_t_out = nn.ModuleList()
        self.embedding_dim = embedding_dim
        self.t = t
        num_heads = [2,4,8,16]
        self.num_layers = num_layers
        dim = dim[::-1]
        self.patch_size = 15
        self.convs = nn.ModuleList()
        
        # test strong fusion
        self.ratio_fusio = ratio_fusio
        if ratio_fusio:

            self.fusion1 = PagFM(embedding_dim,embedding_dim)
            self.fusion2 = PagFM(embedding_dim,embedding_dim)
            self.fusion3 = PagFM(embedding_dim,embedding_dim)
        
        for idx in range(4):
            self.convs.append(
                    ConvModule(
                    in_channels=dim[idx],
                    out_channels=embedding_dim,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=dict(type='SyncBN', requires_grad=True),
                    act_cfg=dict(type='ReLU'))
                )
                
        # 1/8采样结果
        self.fusion_conv = ConvModule(
                            in_channels=embedding_dim*4,
                            out_channels=embedding_dim,
                            kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True))
        
        # self.conv_t_out.append(Mlp(dim[idx]*t,dim[idx]*4,embedding_dim,need_dw=False))
        pre_isa_blocks = nn.ModuleList()
        cpa_blocks = nn.ModuleList()
        post_isa_blocks = nn.ModuleList()
        
        for _ in range(num_layers):
            pre_isa_blocks.append(
                CATBlock(
                    dim = embedding_dim,
                    num_heads=num_heads[idx],
                    patch_size=self.patch_size,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.1,
                    attn_type='ipsa',
                    rpe=False
                )
            )
            cpa_blocks.append(
                CATBlock(
                    dim = embedding_dim,
                    num_heads=1,
                    patch_size=self.patch_size,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.1,
                    attn_type='cpsa',
                    rpe=False
                )
            )
            post_isa_blocks.append(
                CATBlock(
                    dim = embedding_dim,
                    num_heads=num_heads[idx],
                    patch_size=self.patch_size,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.1,
                    attn_type='ipsa',
                    rpe=False
                )
            )
        self.pre_isa_blocks.append(copy.deepcopy(pre_isa_blocks))
        self.cpa_blocks.append(copy.deepcopy(cpa_blocks))
        self.post_isa_blocks.append(copy.deepcopy(post_isa_blocks))

        self.class_token = Class_Token_Seg3(dim=embedding_dim, num_heads=1,num_classes=num_classes)
        self.trans = TransformerClassToken3(dim=embedding_dim, depth=1, num_heads=4,  num_classes =  num_classes,
                                            trans_with_mlp=True, att_type="SelfAttentionWithTime")
        
    def forward(self, query_frame, supp_frame,t = None):
        """ Forward function.
        query_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
        supp_frame: [B*1*c*h/4*w/4, B*1*c*h/8*w/8, B*1*c*h/16*w/16, B*1*c*h/32*w/32]
        Args:
        """
        start_time=time.time()
        query_frame=query_frame[::-1] #[B*(num_clips-1)*c*h/32*w/32, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/4*w/4]
        supp_frame=supp_frame[::-1]
        out_supp_frame = []
        out_memory_frames = []
        T_pre = query_frame[0].shape[1]
        T_tg = supp_frame[0].shape[1]
        # print(len(supp_frame),supp_frame[0].shape)
        # print(len(query_frame),query_frame[0].shape)
        tg_size = supp_frame[-2].shape[-2:]
        # 保留hypercorre5中PagFM的有效性，这里分尺度进行融合，但是需要注意的是，原型依然是固定的
        for idx in range(len(supp_frame)):
            x = supp_frame[idx].flatten(0,1)
            memory = query_frame[idx].flatten(0,1)
            conv = self.convs[idx]
            out_supp_frame.append(conv(x))
            out_memory_frames.append(conv(memory))  
        
        sup0 = resize(input = out_supp_frame[0],size=tg_size,mode='bilinear',align_corners=False)
        sup1 = resize(input = self.fusion1(out_supp_frame[1],sup0),size=tg_size,mode='bilinear',align_corners=False)
        sup2 = resize(input = self.fusion2(out_supp_frame[2],out_supp_frame[1]),size=tg_size,mode='bilinear',align_corners=False)
        sup3 = resize(input = self.fusion3(out_supp_frame[3],out_supp_frame[2]),size=tg_size,mode='bilinear',align_corners=False)
        out_supp_frame = [sup0,sup1,sup2,sup3]

        mem0 = resize(input = out_memory_frames[0],size=tg_size,mode='bilinear',align_corners=False)
        mem1 = resize(input = self.fusion1(out_memory_frames[1],mem0),size=tg_size,mode='bilinear',align_corners=False)
        mem2 = resize(input = self.fusion2(out_memory_frames[2],out_memory_frames[1]),size=tg_size,mode='bilinear',align_corners=False)
        mem3 = resize(input = self.fusion3(out_memory_frames[3],out_memory_frames[2]),size=tg_size,mode='bilinear',align_corners=False)
        out_memory_frames = [mem0,mem1,mem2,mem3]

        out_supp_frame = self.fusion_conv(torch.cat(out_supp_frame,dim=1)) #[BT,C,H,W] 
        out_memory_frames = self.fusion_conv(torch.cat(out_memory_frames,dim=1)) #[BT,C,H,W]

        memory = out_memory_frames.view(-1,T_pre,out_memory_frames.shape[-3],out_memory_frames.shape[-2],out_memory_frames.shape[-1]) #[B,T,C,H,W]
        src = out_supp_frame.view(-1,T_tg,out_supp_frame.shape[-3],out_supp_frame.shape[-2],out_supp_frame.shape[-1]) #[B,T,C,H,W]

        B,num_clips,C,H,W = memory.shape
        
        for _ in range(self.num_layers):
            self.pre_isa_blocks[0][_].H,self.pre_isa_blocks[0][_].W = H , W
            self.cpa_blocks[0][_].H,self.cpa_blocks[0][_].W = H , W
            self.post_isa_blocks[0][_].H,self.post_isa_blocks[0][_].W = H , W

        src = src.permute(0,1,3,4,2).flatten(0,1) #[B*num_clips,H,W,C]
        output = src.view(-1,H*W,C)
        memory = memory.permute(0,1,3,4,2).flatten(0,1) #[B*num_clips,H,W,C]
        memory = memory.view(-1,H*W,C)
        # print("input",output.shape,memory.shape)
        for _ in range(self.num_layers):
            output = self.pre_isa_blocks[0][_](output,memory)
            # print("pre_isa",output.shape)
            output = self.cpa_blocks[0][_](output) 
            # output = self.cpa_blocks[ii][_](output,memory) # 按理说应该是两个
            # print("cap",output.shape)
            output = self.post_isa_blocks[0][_](output) #[B*T,N,C] 
            # print("post_isa",output.shape)
        # print("intmp:",output.shape,memory.shape,num_clips)
        
        # 不需要这个时间模块，只是需要利用原型串通整个时间
        # output = self.tmp_blocks[0](output,memory,num_clips) #[B*T,N,C]
        # print("tm_out",output.shape)
        output = output.view(B,num_clips,H*W,C).permute(0,1,3,2).view(B,num_clips,C,H,W) #[B,N,T*C] 
        # 可学习的原型
        out_cls_mid, cls_tokens = self.class_token(output)
        out_new = self.trans(output, cls_tokens, out_cls_mid) #bxtxcxhxw
        out_new=(torch.chunk(out_new, num_clips, dim=1))
        out_new=[ii.squeeze(1) for ii in out_new]
        return out_new,out_cls_mid
