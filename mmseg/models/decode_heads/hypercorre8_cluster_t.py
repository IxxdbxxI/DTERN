# method from paca_vit
import copy
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .transformer_module import SelfAttentionLayer, CrossAttentionLayer, FFNLayer
from mmseg.models.utils import SelfAttentionBlockWithTime,FreqFusion,simam_module,RCA,LSKblock,SRPS,SRPS2,kmeans,recompute_cluster_centers
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from einops.layers.torch import Rearrange
# test for ratio fusion
from .fdsf import FDSF,PagFM,SWFG2,FSFM,DU2,GAU
# test for CFFM Cross Refiner
from .cffm_module.cffm_transformer import BasicLayer3d3


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

# 1. 采用加权和的方式 2.纯通道注意力
class Channel_Attention_Add(nn.Module):
    def __init__(self, latent_dim,expansion_ratio=1):
        super(Channel_Attention_Add, self).__init__()
        self._latent_dim = latent_dim
        self.temperature = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, expansion_ratio*latent_dim),
            # nn.BatchNorm1d(expansion_ratio*latent_dim),
            # nn.ReLU(),
            # nn.Linear(expansion_ratio*latent_dim, latent_dim)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x): #[b,t,c,n]
        res = x
        n = x.shape[-1]
        x = rearrange(x,"b t c n-> b t (c n)", t = self._latent_dim)
        _max,_ = x.max(dim=-1)
        _avg = x.mean(dim=-1)
        glb = _max + _avg #[b,t]
        glb = self.mlp(glb)
        attn = F.softmax(glb, dim=1)
        x =  x * attn.unsqueeze(-1)
        out = rearrange(x,"b t (c n) -> b t c n",n=n)
        out = self.gamma * out + res
        return out

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

    def __init__(self, dim, patch_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True,attn_type="ipsa",select = False):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size  # Ph, Pw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rpe = rpe
        self.attn_type = attn_type
        if select:
            if self.attn_type == "ipsa":
                self.select_token = None
            else:
                self.select_channel = None

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
        self.select = select
        
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

        if self.select: #好像只适用于静态
            if self.attn_type == "ipsa":
                if self.select_token is None:
                    self.select_token = nn.Parameter(torch.eye(N,requires_grad=True)).unsqueeze(0).unsqueeze(0).expand(memory.shape[0],memory.shape[1],-1,-1)
            else:
                if self.select_channel is None:
                    self.select_channel = nn.Parameter(torch.eye(C,requires_grad=True)).unsqueeze(0).unsqueeze(0).expand(memory.shape[0],memory.shape[1],-1,-1)
        # 把select用在mask上避免矛盾    
        
        q = self.q(x) # [B, T, N, C]
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).unsqueeze(1) # nP*B, nH, N, C
        kv = self.kv(memory) ## [B, T, N, 2C]

        if self.select:
            if self.attn_type == "ipsa":
                kv = self.select_token @ kv
            else:
                kv =  kv @ self.select_channel

        kv=kv.reshape(B_, -1, N, 2, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5) # nP*B, T, nH, N, C
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
        super().__init__()
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
        if self.attn_type == "ipsa": #token selection (top-k or k-means)
            attn = self.attn(patches,patches_memory)  # [B*nP, T, N, C] 
            attn = attn.view(B, Hp//self.patch_size*Wp//self.patch_size,-1, self.patch_size * self.patch_size, C)  # [B,np, T,n,C]
            attn = attn.permute(0,2,1,3,4).contiguous().view(-1, self.patch_size * self.patch_size, C)  # [B*T, nP, patch_size*patch_size, C]
        
        # 第一个ipsa之后进行cpsa,进行全局补充，之后加入时间信息进行交互
        elif self.attn_type == "cpsa": #channel selection  
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


class Cluster_Block(nn.Module):
    def __init__(self, dim,  num_clusters,mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_clusters = num_clusters
        self.norm1 = norm_layer(dim)
        self.Clustering = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=7,stride=1,padding=3),
            nn.GELU(),
            nn.Conv2d(dim,dim,kernel_size=1,stride=1,padding=0),
            nn.GELU(),
            nn.Conv2d(dim,num_clusters,kernel_size=1,stride=1,padding=0,bias=False),
            Rearrange("b c h w -> b c (h w)")
        )
    
    def forward(self,x,H,W): #bt,n,c for all
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W).contiguous()
        return self.Clustering(x)

# 多尺度聚类# 修改为多尺度，多层级聚类,切块让小目标占比更大，交互的时候也需要切块，这里聚类收到的影响较大
class Cluster_Block2(nn.Module):
    def __init__(self, dim,  num_clusters,mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,need_simam=False,
                 use_RCA=False,use_lsk=False,use_SRPS=True,backbone='b1',cityscape=False):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_clusters = num_clusters
        self.norm1 = norm_layer(dim)
        # self.conv1 = nn.Conv2d(dim,dim//4,kernel_size=1)
        self.use_RCA = use_RCA   
        self.use_lsk = use_lsk                 
        self.use_SRPS = use_SRPS     

        self.test_CAM = False
        print("test_CAM:",self.test_CAM)
        if self.test_CAM:
            self.cam = nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1)
            self.Clustering = nn.Sequential(
                    # nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                    # nn.GELU(),
                    nn.Conv2d(dim,num_clusters,kernel_size=1,stride=1,padding=0,bias=False),
                    Rearrange("b c h w -> b c (h w)")
            )

        elif use_RCA:  
            self.RCA = RCA(dim)
            self.Clustering = nn.Sequential(
                    nn.Conv2d(dim,num_clusters,kernel_size=1,stride=1,padding=0,bias=False),
                    Rearrange("b c h w -> b c (h w)")
            )

        elif use_lsk:
            self.lsk = LSKblock(dim) #RF=3、7、11、15
            self.Clustering = nn.Sequential(
                    # nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                    # nn.GELU(),
                    nn.Conv2d(dim,num_clusters,kernel_size=1,stride=1,padding=0,bias=False),
                    Rearrange("b c h w -> b c (h w)")
            )
        elif self.use_SRPS:
            # self.SRPS = SRPS(dim, (15, 10), nn.BatchNorm2d, 11) #(30,15) (20,12) (15,10)
            dw_kernel_size = [11,9,7,5]
            no_local = False
            no_global = False
            print("no_local:",no_local,",no_global:",no_global)
            if backbone == 'b0' or cityscape:
                dw_kernel_size= [13,11,9,7]
            self.SRPS = SRPS2(dim, (15, 10), nn.BatchNorm2d, 11,dw_kernel_size=dw_kernel_size,no_local=no_local,no_global=no_global) #(30,15) (20,12) (15,10)
            self.Clustering = nn.Sequential(
                    # nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                    # nn.GELU(),
                    nn.Conv2d(dim,num_clusters,kernel_size=1,stride=1,padding=0,bias=False),
                    Rearrange("b c h w -> b c (h w)")
            )
        else:
            patch_size = [15,7,3,1] # [21,15,11,7] [15,11,7,3] [11,7,3,1]
            pool_size = [1,3,5,5]

            self.branchs = nn.ModuleList()
            self.pool_xs = nn.ModuleList()
            self.pool_ys = nn.ModuleList()
            
            for pool in pool_size:
                self.pool_xs.append(nn.AvgPool2d(kernel_size=(pool,1),stride=(1,1),padding=(pool//2,0)))
                self.pool_ys.append(nn.AvgPool2d(kernel_size=(1,pool),stride=(1,1),padding=(0,pool//2)))
            
            self.conv_pool = nn.Conv2d(dim,dim,kernel_size=1)

            # block2
            for patch in patch_size:
                # self.branchs.append(nn.Sequential(
                #     nn.Conv2d(dim,dim//4,kernel_size=patch,stride=1,padding=patch//2,groups=dim//4),
                #     nn.GELU(),
                #     nn.Conv2d(dim//4,dim//4,kernel_size=1,stride=1,padding=0),
                #     nn.GELU()
                # ))
                # 修复dw + pw
                self.branchs.append(nn.Sequential(
                    nn.Conv2d(dim,dim,kernel_size=patch,stride=1,padding=patch//2,groups=dim),
                    nn.GELU(),
                    nn.Conv2d(dim,dim//4,kernel_size=1,stride=1,padding=0),
                    nn.GELU()
                ))

            self.Clustering = nn.Sequential(
                    nn.Conv2d(2*dim,num_clusters,kernel_size=1,stride=1,padding=0,bias=False),
                    Rearrange("b c h w -> b c (h w)")
            )
            self.need_simam = need_simam
            if need_simam:
                self.simam = simam_module()

        # block2_2的,添加了原始的7x7卷积，再更改了Clustering
        # self.branchs.append(nn.Sequential(
        #         nn.Conv2d(dim,dim,kernel_size=7,stride=1,padding=3,groups=dim),
        #         nn.GELU(),
        #         nn.Conv2d(dim,dim,kernel_size=1,stride=1,padding=0),
        #         nn.GELU()
        #     ))

        # self.Clustering = nn.Sequential(
        #         nn.Conv2d(2*dim,dim,kernel_size=1,stride=1,padding=0),
        #         nn.GELU(),
        #         nn.Conv2d(dim,num_clusters,kernel_size=1,stride=1,padding=0,bias=False),
        #         Rearrange("b c h w -> b c (h w)")
        # )

    def forward(self,x,H,W): #bt,n,c for all
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W).contiguous()
        outs = []

        if self.test_CAM:
            x = self.cam(x)
            out = self.Clustering(x)
            return out

        if not self.use_RCA and not self.use_lsk and not self.use_SRPS:
            # block2
            outs.append(x)        
            for i in range(len(self.pool_xs)):
                pool_x = self.pool_xs[i](x)
                pool_y = self.pool_ys[i](x)
                # print("pool_out",pool_x.shape,pool_y.shape)
                pool_xy = pool_x + pool_y + x
                pool_xy = self.conv_pool(pool_xy)
                outs.append(self.branchs[i](pool_xy))
            out = torch.cat(outs,dim=1)

            if self.need_simam:
                out = self.simam(out)

        elif self.use_RCA:
            out = self.RCA(x)
        
        elif self.use_lsk:
            out = self.lsk(x)
        elif self.use_SRPS:
            out = self.SRPS(x)
        return self.Clustering(out)

class Cluster_layer(nn.Module):
    def __init__(self, dim, num_heads,  num_clusters,mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,cluster_with_t=False,t=3,inner_center = True,use_kmeans = False,backbone='b1',cityscape=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.norm1 = norm_layer(dim)
        self.inner_center = inner_center
        self.use_kmeans = use_kmeans # only test kmeans

        self.test_SAM = False

        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        # self.clustering = Cluster_Block(dim,num_clusters,mlp_ratio=mlp_ratio)

        self.clustering = Cluster_Block2(dim,num_clusters,mlp_ratio=mlp_ratio,backbone=backbone,cityscape=cityscape) # 

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*4, act_layer=act_layer, drop=drop)
        self.prompt1 = torch.nn.parameter.Parameter(torch.randn(num_clusters, requires_grad=True)) 
        self.top_down_transform1 = torch.nn.parameter.Parameter(torch.eye(num_clusters), requires_grad=True)
        self.prompt2 = torch.nn.parameter.Parameter(torch.randn(num_clusters, requires_grad=True)) 
        self.top_down_transform2 = torch.nn.parameter.Parameter(torch.eye(num_clusters), requires_grad=True)
        self.cluster_with_t = cluster_with_t # 是否用t聚合
        if self.cluster_with_t:
            print("using cluster_with_t by channel attention, t:{}".format(t+1)) 
            self.fusion = Channel_Attention_Add(t+1)

    def forward(self,x,H,W,z=None,mem=None,t=3):
        '''
            x:[b,n,c]
            mem:[bt,n,c]
            z:[b,num_clusters,tn] # 选择迭代更新
        '''
        res = x
        assigned_results = None
        if mem is not None:
            x = torch.cat([mem.view(-1,t,mem.shape[-2],mem.shape[-1]),x.unsqueeze(1)],dim=1) #[b,t,n,c]
            x = x.flatten(0,1) #[bt,n,c]
            t = t+1
        else:
            t = 1
        
        # only test kmeans
        if self.use_kmeans:
            dim = x.shape[-1]
            cluster_x_z = None
            cluster_x = rearrange(x,"(b t) n c -> b (t n) c",b=res.shape[0])
            pesudo_labels_batch = []
            for i in range(cluster_x.shape[0]):
                # kmeans
                b_cluster = cluster_x[i].view(-1,dim).clone().detach().cpu().numpy() # [tn,c]
                indexes_for_clusters = do_clustering(b_cluster, self.num_clusters,1)
                pesudo_labels_batch.append(indexes_for_clusters.view(1,-1)) #[1,tn]

            pesudo_labels_batch = torch.cat(pesudo_labels_batch, dim=0) #[b,tn] 这里可能需要detach()
            cluster_centers_batch = recompute_cluster_centers(cluster_x, pesudo_labels_batch, self.num_clusters)
            C_in = cluster_centers_batch
            # print("C_In",C_in.shape)

        else:
            # clustering
            cluster_x_z = self.clustering(x,H,W) # [bt,num_clusters,n]
            cluster_x_z = rearrange(cluster_x_z,"(b t) c n -> b t c n",b=res.shape[0])
            # 考虑融合所有的t构建一个高置信度引导的情况,学习一个(b,bum_clusters,n) 计算cluster_loss的基础上添加high confidence loss(KL_loss(fusion_y,target_y))
            # 没有利用互补信息(CA on time)
            # 这里添加t上的交互
            # print("fusion:cluster_x_z",cluster_x_z.shape)
            if self.cluster_with_t:
                cluster_x_z = self.fusion(cluster_x_z) 
            # print("fusion out",cluster_x_z.shape)
            cluster_x_z = cluster_x_z.permute(0,2,1,3).contiguous().flatten(2) # [b,num_clusters,tn]

            if self.test_SAM:
                # # inner_time_cluster + cross_time_cluster   
                center = rearrange(cluster_x_z,"b c (t n) -> b t c n",t = t)
                center = center.softmax(dim=-1) # [b,t,num_clusters,n]
                center_x = rearrange(x,"(b t) n c -> b t n c",t=t)
                center = center @ center_x #[b,t,num_clusters,c]
                C_in = center[:,-1] + center[:,:-1].sum(dim=1) # 增强表示

                # C_in = center.average(dim=1) # 增强表示
            else:
                
                if z is not None:
                    # token select 表示的是前后层选择性的对每个聚类对应的区域进行调整
                    # channel select 是选择性的对每个区域对应的聚类分配权重进行调整  Adjust_Cluster_And_Token
                    z = rearrange(z,'b c n -> b n c') #(b,tn,num_clusters)
                    cluster_x_z = rearrange(cluster_x_z,'b c n -> b n c') #(b,tn,num_clusters)
                    # select:
                    cos_sim = F.normalize(z, dim=-1) @ F.normalize(self.prompt1[None, ..., None], dim=1)  # B, N, 1 
                    mask = cos_sim.clamp(0, 1)
                    z = z * mask    #token scaled dot product(对每个token的不同channel同等缩放，对不同的token不同等缩放)
                    z = z @ self.top_down_transform1 #(对不同token的同一channel同等缩放，对不同channle不同缩放)
                    # select:
                    cos_sim = F.normalize(cluster_x_z, dim=-1) @ F.normalize(self.prompt2[None, ..., None], dim=1)  # B, N, 1 
                    mask = cos_sim.clamp(0, 1)
                    cluster_x_z = cluster_x_z * mask
                    cluster_x_z = cluster_x_z @ self.top_down_transform2
                    cluster_x_z = (cluster_x_z + z)/2.
                    cluster_x_z = rearrange(cluster_x_z,"b n c -> b c n")

                if self.inner_center:
                    # # inner_time_cluster + cross_time_cluster   
                    center = rearrange(cluster_x_z,"b c (t n) -> b t c n",t = t)
                    assigned_results = center
                    center = center.softmax(dim=-1) # [b,t,num_clusters,n]
                    center_x = rearrange(x,"(b t) n c -> b t n c",t=t)
                    center = center @ center_x #[b,t,num_clusters,c]
                    cos_sim = torch.sigmoid( 
                        (self.sim_beta + self.sim_alpha * F.cosine_similarity(center[:,-1].unsqueeze(1),center[:,:-1],dim=-1))
                    ) #[b,t,num_cluster]
                    C_in = center[:,-1] + (cos_sim.unsqueeze(-1) * center[:,:-1]).sum(dim=1) # 增强表示
                
                else:
                    cluster_x = cluster_x_z.softmax(dim=-1) # [b,num_clusters,tn] # 在空间上分配的softmax，而不是采一般的聚类分配方式，列表示类性质，行表示标签性质
                    # cluster_x 可以加入对比损失
                    x = rearrange(x,'(b t) n c -> b (t n) c',b=res.shape[0])
                    C_in = cluster_x @ x #vison_token [b,num_clusters,c] 将当前的特征赋予给当前的聚类,而不是根据相似度把聚类分配给特征

        C_in = self.norm1(C_in)
        src = rearrange(res,"b n c -> n b c")
        mem = rearrange(C_in, "b n c -> n b c")
        out,_ = F.multi_head_attention_forward(
            query=src,
            key=mem,
            value=mem,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q.weight,
            k_proj_weight=self.k.weight,
            v_proj_weight=self.v.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q.bias, self.k.bias, self.v.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.attn_drop,
            out_proj_weight=self.proj.weight,
            out_proj_bias=self.proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=not self.training,  # for visualization
            # average_attn_weights=False,
        )
        out = rearrange(out,"n b c -> b n c")
        out = self.proj_drop(out)
        
        out = res + self.norm1(out)
        out = out + self.norm1(self.mlp(out,H,W))
        return out,cluster_x_z,assigned_results
    
# 原型一致化时间,from A Transformer-based Decoder for Semantic Segmentation with Multi-level Context Mining iccv22
# 设计能够有效缓解segformer的尺度拼接问题 # 用于画图,直接改里面参数名称,把Class_Token_Seg3和TransformerClassToken3修改成External_Cluster_decoder
class External_Cluster_decoder(nn.Module):
    def __init__(self, dim, num_heads=8, num_classes=150, qkv_bias=True, qk_scale=None, T=3):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.zeros(1, T, num_classes, dim)) #这里需要T来表示不同帧里面的类别可变性，也像类别原型一样
        
        self.prop_token = nn.Parameter(torch.zeros(1, num_classes, dim)) #可学习的聚类中心，结果根据cls_token结果进行更新产生
        
        # self.prop_token = nn.Parameter(torch.zeros(1, num_classes, dim)) #原型，需要扩张到t的表示原型的一致性串通时间维度,这里可以适当增大原型数目进行测试
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
        
        self.T = T
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.prop_token, std=.02)
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
             
    def forward(self, x):
        b, t, c, h, w = x.size()
        res = x
        assert t == self.T, "Input tensor has wrong time"
        x = x.flatten(3).transpose(-1, -2) #[b,t,n,c]
        B, T, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1, -1) # [B, T, num_classes, dim]
        prop_tokens = self.prop_token.unsqueeze(1).expand(B, T, -1, -1) # [B, T, num_classes, dim]
        
        # x = torch.cat((cls_tokens, x), dim=2) #[B, T, num_classes + N, dim]
        # B, T, N, C = x.shape
        q = self.q(x).reshape(B, T, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = self.k(cls_tokens).unsqueeze(1).reshape(B, T, self.num_classes, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4) #[B, T, num_heads, num_classes, dim]
        
        k = k * self.scale
        attn = (k @ q.transpose(-2, -1)).squeeze(2).transpose(-2, -1) # [B, T, N, num_classes]
        # attn = attn[:,:, self.num_classes:] #[B, T, N, num_classes] 这里其实不需要拼接起来
        
        out_cls_mid = attn.permute(0, 1, 3, 2).reshape(b, t, -1, h, w)
        # x_cls,prop:out_cls_mid, cls_tokens
        x_cls = out_cls_mid.flatten(3).transpose(-1, -2)   # [b,t,n,c]
        x_cls = x_cls.softmax(dim=-1)
        cls = x_cls @ prop_tokens
        cls = cls.permute(0, 1, 3, 2).reshape(b, t, c, h, w) 
        x2 = self.cross_attn(res, cls)
        # 这里减小了mIoU,但是增加了VSC连续性，就是过度修复了连续性，但是影响了miou
        out = res + x2 # 这里x2的影响太大了,相较于一般情况，out_cls_mid的损失可以降低权重，因为x2减小了影响 0.42 
        # out = res + 0.5 * x2    # loss: 0.25
        # out = res + 0.8 * x2    # loss: 0.4
        # out = res + 0.2 * x2    # loss: 0.1
        return out_cls_mid,out


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
        
        # test 这里的原型应该产生于类别，还是类别产生于原型，还是互相不影响,当数目增加的时候需要加上PagFM才能提升效果，但是数目减小加PagFM会降低效果
        self.prop_token = nn.Parameter(torch.zeros(1, num_classes, dim)) #原型，需要扩张到t的表示原型的一致性串通时间维度,这里可以适当增大原型数目进行测试
        
        # self.prop_token = nn.Parameter(torch.zeros(1, num_classes, dim)) #原型，需要扩张到t的表示原型的一致性串通时间维度,这里可以适当增大原型数目进行测试
        
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
        attn = attn[:,:, self.num_classes:] #[B, T, N, num_classes] 这里其实不需要拼接起来
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

        # test 原型数目,数目增加需要加PagFM
        # self.num_prototypes = 26+num_classes

        # self.linear_cls = nn.Linear(num_classes,self.num_prototypes)

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
        # x1 = self.attn(x, x)
        
        #cross images attention
        # out_cls_mid = self.linear_cls(out_cls_mid) #b,t,n,num_classes -> b,t,n,num_prototypes,这里的映射好像不太合适，需要修改调整,这里需要询问一下师兄

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

# 将聚类贯穿到底
class hypercorre_topk2(nn.Module):
    """ top-k2: same selections for each reference image so that attention decoder can be used

    Args:
    num_feats: number of features being used

    """

    def __init__(self,dim=[64, 128, 320, 512], num_layers=1, t=3, time_decoder_layer=3,embedding_dim=256,num_classes = 124,
                 cross_method='CAT',ratio_fusio = False,num_clusters=150,cluster_with_t=False,ratio_max = False,
                 backbone='b1',cityscape=False,need_segdeformer=True):
        super().__init__()
        dim = dim[::-1]
        self.dim=dim
        self.pre_isa_blocks = nn.ModuleList()
        self.cpa_blocks = nn.ModuleList()
        self.post_isa_blocks = nn.ModuleList()
        self.tmp_blocks = nn.ModuleList()
        self.conv_t_out = nn.ModuleList()
        self.embedding_dim = embedding_dim
        num_heads = [2,4,8,16]
        self.num_layers = num_layers
        self.patch_size = 15
        self.convs = nn.ModuleList()
        self.sub_convs = nn.ModuleList()
        self.cross_method = cross_method
        self.t = 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.ratio_max = ratio_max
        self.need_segdeformer=need_segdeformer
        # test strong fusion
        self.ratio_fusio = ratio_fusio
        if ratio_fusio and ratio_fusio == "DU":
            # self.fusion1 = PagFM(embedding_dim,embedding_dim)
            # self.fusion2 = PagFM(embedding_dim,embedding_dim)
            # self.fusion3 = PagFM(embedding_dim,embedding_dim)
            # self.fusion1 = SWFG2(embedding_dim,embedding_dim)
            # self.fusion2 = SWFG2(embedding_dim,embedding_dim)
            # self.fusion3 = SWFG2(embedding_dim,embedding_dim)
            # self.fusion1 = FSFM(embedding_dim,embedding_dim)
            # self.fusion2 = FSFM(embedding_dim,embedding_dim)
            # self.fusion3 = FSFM(embedding_dim,embedding_dim)
            self.fusion1 = DU2(embedding_dim)
            self.fusion2 = DU2(embedding_dim)
            self.fusion3 = DU2(embedding_dim)

        if ratio_fusio and ratio_fusio == "GAU":
            self.fusion1 = GAU(dim[0],dim[1])
            self.fusion2 = GAU(dim[1],dim[2])
            self.fusion3 = GAU(dim[2],dim[3])
            self.down_conv = nn.Conv2d(dim[3],dim[3],kernel_size=2,stride=2)

            
        if ratio_fusio and ratio_fusio == "freq_fuse":
            self.freqfusions = nn.ModuleList()
            self.down_conv = nn.Conv2d(sum(dim),sum(dim),kernel_size=2,stride=2)
            pre_c = dim[0]
            for i in range(3):
                c = dim[i+1]
                freqfusion = FreqFusion(
                        hr_channels=c, lr_channels=pre_c, scale_factor=1,
                        feature_resample=True,
                        hamming_window=False,
                        compressed_channels= (pre_c + c) // 4
                    )
                self.freqfusions.append(freqfusion)
                pre_c += c

        else:
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
        
        if self.ratio_fusio == "freq_fuse":
            self.fusion_conv = ConvModule(
                            in_channels=sum(dim),
                            out_channels=embedding_dim,
                            kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True))
        if self.ratio_fusio == "GAU":
            self.fusion_conv = ConvModule(
                            in_channels=dim[3],
                            out_channels=embedding_dim,
                            kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True))

        self.cluster_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.cluster_blocks.append(
                Cluster_layer(
                    dim=embedding_dim,
                    num_heads=num_heads[i],
                    num_clusters=num_clusters,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,t=3,cluster_with_t=cluster_with_t,backbone=backbone,cityscape=cityscape)
            )

        # self.class_token = Class_Token_Seg3(dim=embedding_dim, num_heads=1,num_classes=num_classes,T=self.t)
        # self.trans = TransformerClassToken3(dim=embedding_dim, depth=1, num_heads=4,  num_classes =  num_classes,
        #                                     trans_with_mlp=True, att_type="SelfAttentionWithTime")
        if self.need_segdeformer:
            self.extern_cluster_decoder = External_Cluster_decoder(dim=embedding_dim, num_heads=1, num_classes=num_classes, T=self.t)
        
    def forward(self, query_frame, supp_frame,t = None,img_metas=None):
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
        # 这里必须要去掉冗余才行
        # 测试先resize到最大，再融合下采样，用于先弥补小分辨率的问题
        tg_size = supp_frame[-2].shape[-2:]
        if not self.ratio_fusio:
            k_ratio = -2
            if self.ratio_max:
                k_ratio = -1
            for idx in range(len(supp_frame)):
                x = supp_frame[idx].flatten(0,1)
                memory = query_frame[idx].flatten(0,1)
                conv = self.convs[idx]
                out_supp_frame.append(
                    resize(               # 
                        input=conv(x),
                        size=supp_frame[k_ratio].shape[-2:],
                        mode='bilinear',
                        align_corners=False))
                out_memory_frames.append(
                    resize(
                        input=conv(memory),
                        size=query_frame[k_ratio].shape[-2:], # 1/8
                        mode='bilinear',
                        align_corners=False))
                
        elif self.ratio_fusio == "sub":
                tg_size = supp_frame[-2].shape[-2:]
                for idx in range(len(supp_frame)):
                    x = supp_frame[idx]
                    memory = query_frame[idx]
                    conv = self.convs[idx]
                    # sub_conv = self.sub_convs[idx]

                    out_supp_frame.append(
                        resize(
                            input=conv(x.flatten(0,1)),
                            size=supp_frame[-2].shape[-2:],
                            mode='bilinear',
                            align_corners=False))
                    
                    sub_x = x.repeat(1,T_pre,1,1,1).flatten(0,1)
                    memory = memory.flatten(0,1)
                    sub_mem = sub_x - memory
                    out_memory_frames.append(
                        resize(
                            input=conv(sub_mem * self.sigmoid(self.gap(sub_mem)) + memory), #sub_fusion2
                            # input=conv(torch.sigmoid(sub_conv(sub_x - memory)) * memory), #sub_fusion1
                            size=query_frame[-2].shape[-2:], # 1/8
                            mode='bilinear',
                            align_corners=False))

        elif self.ratio_fusio == "freq_fuse": #解决小分辨率问题
            lowres_feat = torch.cat([query_frame[0],supp_frame[0]],dim=1).flatten(0,1)
            # print('lowres_feat',lowres_feat.shape) 
            for idx in range(len(supp_frame)-1):
                hires_feat = torch.cat([query_frame[idx+1],supp_frame[idx+1]],dim=1).flatten(0,1)
                # print('hires_feat',hires_feat.shape)
                _, hires_feat, lowres_feat = self.freqfusions[idx](hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=False)
                b, _, h, w = hires_feat.shape
                lowres_feat = torch.cat([hires_feat.reshape(b * 4, -1, h, w), 
                                            lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
            out_fuse = lowres_feat
            # print("out_fuse",out_fuse.shape)

        elif self.ratio_fusio == "GAU":
            sup0 = self.fusion1(supp_frame[0].flatten(0,1),supp_frame[1].flatten(0,1))
            sup1 = self.fusion2(sup0,supp_frame[2].flatten(0,1))
            sup2 = self.fusion3(sup1,supp_frame[3].flatten(0,1))
            out_supp_frame = sup2

            mem0 = self.fusion1(query_frame[0].flatten(0,1),query_frame[1].flatten(0,1))
            mem1 = self.fusion2(mem0,query_frame[2].flatten(0,1))
            mem2 = self.fusion3(mem1,query_frame[3].flatten(0,1))
            out_memory_frames = mem2

        else:
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


        if self.ratio_fusio == "freq_fuse":
            out_fuse = self.down_conv(out_fuse)
            out_fuse = self.fusion_conv(out_fuse)
            out_fuse = rearrange(out_fuse,'(b t) c h w -> b t c h w',t=T_pre+T_tg)
            out_memory_frames = out_fuse[:,:T_pre]
            out_supp_frame = out_fuse[:,T_pre:]

        elif self.ratio_fusio == "GAU":
            out_supp_frame = self.down_conv(out_supp_frame)
            out_memory_frames = self.down_conv(out_memory_frames)
            out_supp_frame = self.fusion_conv(out_supp_frame)
            out_memory_frames = self.fusion_conv(out_memory_frames)

        else:
            out_supp_frame = self.fusion_conv(torch.cat(out_supp_frame,dim=1)) #[BT,C,H,W] 
            out_memory_frames = self.fusion_conv(torch.cat(out_memory_frames,dim=1)) #[BT,C,H,W]

        out_supp_frame = resize(input=out_supp_frame,size=tg_size,mode='bilinear',align_corners=False)
        out_memory_frames = resize(input=out_memory_frames,size=tg_size,mode='bilinear',align_corners=False)
        memory = out_memory_frames.view(-1,T_pre,out_memory_frames.shape[-3],out_memory_frames.shape[-2],out_memory_frames.shape[-1]) #[B,T,C,H,W]
        src = out_supp_frame.view(-1,T_tg,out_supp_frame.shape[-3],out_supp_frame.shape[-2],out_supp_frame.shape[-1]) #[B,T,C,H,W]

        B,_,C,H,W = memory.shape
        z = None


        src = rearrange(src,'b t c h w -> (b t) c (h w)')
        memory = rearrange(memory,'b t c h w -> (b t) c (h w)')
        src = src.permute(0,2,1)
        memory = memory.permute(0,2,1)
        mem_out=None
        for idx in range(self.num_layers):
            if idx == 0:
                x,z,assigned_results = self.cluster_blocks[idx](src, H=H, W=W, mem = memory)
            elif idx == 1:
                x,z,assigned_results = self.cluster_blocks[idx](x, H=H, W=W, z=z, mem = memory) #聚类学习[b,n,c]
            else:
                x,_,_ = self.cluster_blocks[idx](x, H=H, W=W) # 自身的加强
            print("layer{idx}: assigned_results",assigned_results.shape)
        
        # z: cluster # [b,num_clusters,tn] assigned_results:[b,t,num_clusters,n]
        # print("z",z.shape)
        if z is not None:
            z = rearrange(z,'b c (t h w) -> b t c h w',t=T_pre+T_tg,h=H,w=W)
            # z = rearrange(z,'b c t (h w) -> b t c h w',h=H,w=W)
        x = rearrange(x, '(b t) (h w) c -> b t c h w', b=B, t=T_tg,h=H,w=W)
        # 同Focal,只有tg得到更新
        if self.need_segdeformer:
            # 可学习的原型 (B,num_clips,C,H,W)
            # out_cls_mid, cls_tokens =  self.class_token(x)
            # out_new = self.trans(x, cls_tokens, out_cls_mid) #bxtxcxhxw

            out_cls_mid,out_new = self.extern_cluster_decoder(x)
        
            out_new=(torch.chunk(out_new, T_tg, dim=1))
            out_new=[ii.squeeze(1) for ii in out_new] # focal情况下就是对所有帧的分割细化(还对目标帧进行了更新)，CAT就是对目标帧的分割细化
            return out_new,out_cls_mid.squeeze(1),z,mem_out,assigned_results #[b,c,h,w]
        else:
            out_new=(torch.chunk(x, T_tg, dim=1))
            out_new=[ii.squeeze(1) for ii in out_new] # focal情况下就是对所有帧的分割细化(还对目标帧进行了更新)，CAT就是对目标帧的分割细化
            return out_new,z,assigned_results
