# 全部往paca-vit更改
import copy
from einops import rearrange
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
from einops.layers.torch import Rearrange
# test for ratio fusion
from .fdsf import FDSF,PagFM,SWFG2,FSFM

def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
    x1 = F.normalize(x1,dim=-1)
    x2 = F.normalize(x2,dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

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
    
class Channel_Attention(nn.Module):
    def __init__(self, latent_dim,expansion_ratio=4):
        super(Channel_Attention, self).__init__()
        self._latent_dim = latent_dim
        self.temperature = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, expansion_ratio*latent_dim),
            nn.BatchNorm1d(expansion_ratio*latent_dim),
            nn.ReLU(),
            nn.Linear(expansion_ratio*latent_dim, latent_dim)
        )
        

    def forward(self, x,t): #[b,tn,c]
        x = rearrange(x,"b (t n) c -> b t (n c)", t = self._latent_dim)
        _max,_ = x.max(dim=-1)
        _avg = x.mean(dim=-1)
        glb = _max + _avg #[b,t]
        glb = self.mlp(glb)
        attn = F.softmax(glb, dim=1)
        out = x + x * attn.unsqueeze(-1) 
        return out

class Channel_Attention_Add(nn.Module):
    def __init__(self, latent_dim,expansion_ratio=4):
        super(Channel_Attention_Add, self).__init__()
        self._latent_dim = latent_dim
        self.temperature = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, expansion_ratio*latent_dim),
            nn.BatchNorm1d(expansion_ratio*latent_dim),
            nn.ReLU(),
            nn.Linear(expansion_ratio*latent_dim, latent_dim)
        )
        

    def forward(self, x): #[b,tn,c]
        c = x.shape[-1]
        x = rearrange(x,"(b t) n c -> b t (n c)", t = self._latent_dim)
        _max,_ = x.max(dim=-1)
        _avg = x.mean(dim=-1)
        glb = _max + _avg #[b,t]
        glb = self.mlp(glb)
        attn = F.softmax(glb, dim=1)
        x =  x.transpose(1,2) @ attn.unsqueeze(-1) #b  (n c) t @ b t 1 -> b (n c) 1
        out = rearrange(x,"b (n c) 1 -> b n c",c=c)
        return out




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


class Cluster_Block(nn.Module):
    def __init__(self, dim,  num_clusters,mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_clusters = num_clusters
        self.norm1 = norm_layer(dim)
        self.Clustering = nn.Sequential( #符合语义聚类
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

class Cluster_layer(nn.Module):
    def __init__(self, dim, num_heads,  num_clusters,mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,cluster_with_t = False,t=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.norm1 = norm_layer(dim)
        self.clustering = Cluster_Block(dim,num_clusters,mlp_ratio=mlp_ratio)
        self.conv_sikp = nn.Conv2d(dim,dim,1) 
        self.conv1 = nn.Conv2d(dim,dim,1)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*4, act_layer=act_layer, drop=drop)
        # select token and select channels for cluster_fusion
        self.prompt1 = torch.nn.parameter.Parameter(torch.randn(dim, requires_grad=True)) 
        self.top_down_transform1 = torch.nn.parameter.Parameter(torch.eye(dim), requires_grad=True)
        self.prompt2 = torch.nn.parameter.Parameter(torch.randn(dim, requires_grad=True)) 
        self.top_down_transform2 = torch.nn.parameter.Parameter(torch.eye(dim), requires_grad=True)
        
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        
        self.cluster_with_t = cluster_with_t # 是否用t聚合
        if self.cluster_with_t:
            print("using cluster_with_t by channel attention, t:{}".format(t+1)) 
            self.fusion = Channel_Attention_Add(t+1)

    def forward(self,x,H,W,z=None,mem=None,t=3):
        '''
            x:[b,n,c]
            mem:[bt,n,c]
            z:[bt,num_clusters,c] # 选择迭代更新
        '''
        res = x
        b,n,c = x.shape
        assert n == H*W, "input feature has wrong size"

        feature = self.conv_sikp(x.transpose(1,2).view(b,c,H,W))
        feature = feature.flatten(2).transpose(1,2) 

        x = self.conv1(x.transpose(1,2).view(b,c,H,W))
        x = x.flatten(2).transpose(1,2)
        tg_x = x
        # print("in_put",x.shape,mem.shape)
        if mem is not None:
            mem = self.conv1(mem.transpose(1,2).view(b*t,c,H,W))
            mem = mem.flatten(2).transpose(1,2)

            x = torch.cat([mem.view(-1,t,mem.shape[-2],mem.shape[-1]),x.unsqueeze(1)],dim=1) #[b,t,n,c]
            x = x.flatten(0,1) #[bt,n,c]
            t = t+1
        else:
            t = 1
            # print("x",x.shape) #x torch.Size([2, 14400, 256])
        # clustering
        cluster_x_z = self.clustering(x,H,W) # [bt,num_clusters,n]
        
        cluster_x_z = cluster_x_z.softmax(dim=-1) # [bt,num_clusters,n] # 在空间上分配的softmax，而不是采一般的聚类分配方式，列表示类性质，行表示标签性质
        # cluster_x = self.softmax(self.sigmoid(cluster_x_z)/self.temperature) # [b,num_clusters,tn] # 在空间上分配的softmax，而不是采一般的聚类分配方式，列表示类性质，行表示标签性质
        
        # x = rearrange(x,'(b t) n c -> b (t n) c',b=res.shape[0])
        # 一个是(b cluster,tn) @ (b tn C) = (b cluster C) 每个cluster用了所有的tn加权， 
        
        cluster_x_z = cluster_x_z @ x #vison_token [bt,num_clusters,c] 将当前的特征赋予给当前的聚类,而不是根据相似度把聚类分配给特征
        center  = cluster_x_z
        # 先融合再更新，需要在t方向上加权求和tc,tc->c,增加时间一致性

        if self.cluster_with_t:
            cluster_x_z = self.fusion(cluster_x_z) # [b,num_clusters,c]
        else:
            cluster_x_z = rearrange(cluster_x_z,"(b t) n c -> b (t n) c",b=res.shape[0])

        if z is not None:
        # 提供一个融合的模块，来完成cluster的更新，,进行token_selection然后重组 (更新相似度高的cluster？ or CA(cluster_x_z,mem_cluster))从memory中获取信息
          # 先select 再求平均(k_means) 聚类中心会不会因此偏移向tg的方向
            # select
            cos_sim = F.normalize(z, dim=-1) @ F.normalize(self.prompt1[None, ..., None], dim=1)  # B, N, 1
            mask = cos_sim.clamp(0, 1)
            z = z * mask
            z = z @ self.top_down_transform1
            # select:
            cos_sim = F.normalize(cluster_x_z, dim=-1) @ F.normalize(self.prompt2[None, ..., None], dim=1)  # B, N, 1
            mask = cos_sim.clamp(0, 1)
            cluster_x_z = cluster_x_z * mask
            cluster_x_z = cluster_x_z @ self.top_down_transform2
            cluster_x_z = (cluster_x_z + z)/2.
        C_in = cluster_x_z  # [b,num_clusters,c]

# --------------------------------------------------以下代码是从cluster-former向paca-vit方向的合并-----------------------------
        '''1.接下来是特征聚合,这里只聚合来自target的特征是不合理的,因为这样会导致特征的丢失,所以这里需要将来自source的特征也聚合进来
           但是这里的聚合为什么要选择使用最原始的特征的加权和,而不是采用得到的有效的聚类token C_in,对这一段直接删除可以得到paca-vit的结果,也可以测试一下这里的全新聚类
        '''
        # x = rearrange(x,"(b t) n c -> b (t n) c",b=res.shape[0])
        # #  (b,num_clusters,c) (b,tn,c) -> (b,num_clusters,tn)
        # similarity = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(C_in.type(torch.float), x.type(torch.float)))
        # # 聚合所有的情况
        # _, max_idx = similarity.max(dim=1, keepdim=True)
        # mask = torch.zeros_like(similarity)
        # mask.scatter_(1, max_idx, 1.)
        # similarity= similarity*mask #(b,num_clusters,n)
        # # (b,1,n,c) (b,num_clusters,n,1) -> (b,num_clusters,n,c) -> (b,num_clusters,c) 下面采用余弦相似度分配，但是聚合是按照欧式距离的均值聚合
        # out = ((x.unsqueeze(dim=1)*similarity.unsqueeze(dim=-1) ).sum(dim=2) + C_in)/ (mask.sum(dim=-1,keepdim=True)+ 1.0) 
        # # 重新得到一组新的聚类中心，特征来自于原始tn的特征的均值
        
        # # 计算更新的全部聚类中心向target的分配
        # similarity_tg = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(C_in.type(torch.float), feature.type(torch.float)))
        # _, max_idx = similarity_tg.max(dim=1, keepdim=True)
        # mask = torch.zeros_like(similarity_tg)
        # mask.scatter_(1, max_idx, 1.)
        # similarity_tg= similarity_tg*mask #(b,num_clusters,n)
        # #  (b,num_clusters,1,c) * (b,num_clusters,n,1) -> (b,num_clusters,n,c)
        # out = (out.unsqueeze(dim=2)*similarity_tg.unsqueeze(dim=-1)).sum(dim=1) # b,n,c， 
#-----------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------only target agreement clusterformer--------------------------------------------------
        # '''
        #     2.这里只采用了target进行聚类中心的聚合,从参数上分析,这里应该加上用于对齐的q(C_in), k(tg_c), v(feature)
        # '''
        # #  (b,num_clusters,c) (b,n,c) -> (b,num_clusters,n)
        # similarity = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(C_in.type(torch.float), tg_x.type(torch.float))) #测试了一组feature
        # # aggregate feature avg
        # _, max_idx = similarity.max(dim=1, keepdim=True)
        # mask = torch.zeros_like(similarity)
        # mask.scatter_(1, max_idx, 1.)
        # similarity= similarity*mask #(b,num_clusters,n)
        # # (b,1,n,c) (b,num_clusters,n,1) -> (b,num_clusters,n,c) -> (b,num_clusters,c)
        # out = ((feature.unsqueeze(dim=1)*similarity.unsqueeze(dim=-1) ).sum(dim=2) + C_in)/ (mask.sum(dim=-1,keepdim=True)+ 1.0) 
        # # 分配
        # #  (b,num_clusters,1,c) * (b,num_clusters,n,1) -> (b,num_clusters,n,c)
        # out = (out.unsqueeze(dim=2)*similarity.unsqueeze(dim=-1)).sum(dim=1) # b,n,c， 这也是相当于从聚类中心得到加权结果
#----------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------只在乎分配,这里会产生多个点一样的特征的情况------------------------------------------------------------
        '''
            3.遵循paca-vit的内容,得到的C_in就是一组合适的语义聚类中心,无需进行其余的操作,直接分配即可,但是paca-vit的过程保持着对所有聚类中心的加权,而这里只在乎分配
        '''
        # similarity = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(feature.type(torch.float), C_in.type(torch.float)))
        # # aggregate feature avg
        # _, max_idx = similarity.max(dim=2, keepdim=True)
        # mask = torch.zeros_like(similarity)
        # mask.scatter_(2, max_idx, 1.)
        # similarity= similarity*mask #(b,n,n_cluster)
        # # (b,n,n_cluster)(b,n_cluster,c) -> (b,n,c)
        # out = similarity @ C_in  
# ---------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------paca-vit的情况------------------------------------------------------------
        '''
        4. 完全采纳capa-vit的方式,但是相较于paca-vit获得C_in的过程,这里缺乏全局t上的交互,paca-vit得到的C_in通过的tn个点的加权,而这里只有t个(n个点的加权),可以后续考虑融合t
        '''
        # # (b,n,num_clusters)
        similarity = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(feature.type(torch.float), C_in.type(torch.float)))
        # (b,n,num_clusters) @ (b,n,c) -> (b,n,c) 
        out = similarity @ C_in
# ---------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------采用余弦相似度聚合，而不是采用欧式距离的均值聚合------------------------------------------------------------
        '''
        5. 从1,2中选择效果更好的做余弦相似度聚合,结果和欧式距离计算结果是一致的 https://zhuanlan.zhihu.com/p/380389927
        '''
# ---------------------------------------------------------------------------------------------------------------------------------------

        out = self.proj_drop(out)
        out = res + self.norm1(out)
        out = out + self.norm1(self.mlp(out,H,W))
        return out,center,cluster_x_z
    
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

# 将聚类贯穿到底
class hypercorre_topk2(nn.Module):
    """ top-k2: same selections for each reference image so that attention decoder can be used

    Args:
    num_feats: number of features being used

    """

    def __init__(self,dim=[64, 128, 320, 512], num_layers=1, t=3, time_decoder_layer=3,embedding_dim=256,
                 num_classes = 124,cross_method='CAT',ratio_fusio = False,num_clusters=150,cluster_with_t=False,need_segdeformer=False):
        super().__init__()
        self.dim=dim
        self.pre_isa_blocks = nn.ModuleList()
        self.cpa_blocks = nn.ModuleList()
        self.post_isa_blocks = nn.ModuleList()
        self.tmp_blocks = nn.ModuleList()
        self.conv_t_out = nn.ModuleList()
        self.embedding_dim = embedding_dim
        num_heads = [2,4,8,16]
        self.num_layers = num_layers
        dim = dim[::-1]
        self.patch_size = 15
        self.convs = nn.ModuleList()
        self.cross_method = cross_method
        self.tg = 1
        self.tm = 3

        # test strong fusion
        self.ratio_fusio = ratio_fusio
        if ratio_fusio:
            self.fusion1 = PagFM(embedding_dim,embedding_dim)
            self.fusion2 = PagFM(embedding_dim,embedding_dim)
            self.fusion3 = PagFM(embedding_dim,embedding_dim)
            # self.fusion1 = SWFG2(embedding_dim,embedding_dim)
            # self.fusion2 = SWFG2(embedding_dim,embedding_dim)
            # self.fusion3 = SWFG2(embedding_dim,embedding_dim)
            # self.fusion1 = FSFM(embedding_dim,embedding_dim)
            # self.fusion2 = FSFM(embedding_dim,embedding_dim)
            # self.fusion3 = FSFM(embedding_dim,embedding_dim)
        
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
                    norm_layer=nn.LayerNorm,t=self.tm,cluster_with_t=cluster_with_t)
            )

        # 由于分配的聚类方式是不是不需要decoder
        self.need_segdeformer = need_segdeformer
        if need_segdeformer:
            self.class_token = Class_Token_Seg3(dim=embedding_dim, num_heads=1,num_classes=num_classes,T=self.tg)
            self.trans = TransformerClassToken3(dim=embedding_dim, depth=1, num_heads=4,  num_classes =  num_classes,
                                            trans_with_mlp=True, att_type="SelfAttentionWithTime")
        
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
        if not self.ratio_fusio:
            # 尺度交互变成了拼接,依托于后面强大的分辨器，这里可以考虑采用PagFM(在decoder中已经测试有效)
            for idx in range(len(supp_frame)):
                x = supp_frame[idx].flatten(0,1)
                memory = query_frame[idx].flatten(0,1)
                conv = self.convs[idx]
                out_supp_frame.append(
                    resize(
                        input=conv(x),
                        size=supp_frame[-2].shape[-2:],
                        mode='bilinear',
                        align_corners=False))
                out_memory_frames.append(
                    resize(
                        input=conv(memory),
                        size=query_frame[-2].shape[-2:], # 1/8
                        mode='bilinear',
                        align_corners=False))
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
                
        out_supp_frame = self.fusion_conv(torch.cat(out_supp_frame,dim=1)) #[BT,C,H,W] 
        out_memory_frames = self.fusion_conv(torch.cat(out_memory_frames,dim=1)) #[BT,C,H,W]

        memory = out_memory_frames.view(-1,T_pre,out_memory_frames.shape[-3],out_memory_frames.shape[-2],out_memory_frames.shape[-1]) #[B,T,C,H,W]
        src = out_supp_frame.view(-1,T_tg,out_supp_frame.shape[-3],out_supp_frame.shape[-2],out_supp_frame.shape[-1]) #[B,T,C,H,W]

        B,_,C,H,W = memory.shape
        z = None

        src = rearrange(src,'b t c h w -> (b t) c (h w)')
        memory = rearrange(memory,'b t c h w -> (b t) c (h w)')
        src = src.permute(0,2,1)
        memory = memory.permute(0,2,1)
        for idx in range(self.num_layers):
            if idx == 0:
                x,center,z = self.cluster_blocks[idx](src, H=H, W=W, mem = memory)
            elif idx == 1:
                x,center,z = self.cluster_blocks[idx](x, H=H, W=W, z=z, mem = memory) #聚类学习[b,n,c]
            else:
                x,center,_ = self.cluster_blocks[idx](x, H=H, W=W) # 自身的加强
        
        # center: cluster # [bt,num_clusters,c]
        center = rearrange(center, '(b t) n c -> b t n c', b=B, t=T_tg+T_pre) # 用于对比损失
        # z = rearrange(z,'b c t (h w) -> b t c h w',h=H,w=W)
        x = rearrange(x, '(b t) (h w) c -> b t c h w', b=B, t=T_tg,h=H,w=W)
        # 同Focal,只有tg得到更新
        # 可学习的原型 (B,num_clips,C,H,W)
        if self.need_segdeformer:
            out_cls_mid, cls_tokens =  self.class_token(x)
            out_new = self.trans(x, cls_tokens, out_cls_mid) #bxtxcxhxw
            out_new=(torch.chunk(out_new, T_tg, dim=1))
            out_new=[ii.squeeze(1) for ii in out_new] # focal情况下就是对所有帧的分割细化(还对目标帧进行了更新)，CAT就是对目标帧的分割细化
            return out_new,out_cls_mid.squeeze(1),center #[b,c,h,w]

        else:
            out_new = x
            out_new=(torch.chunk(out_new, T_tg, dim=1))
            out_new=[ii.squeeze(1) for ii in out_new] # focal情况下就是对所有帧的分割细化(还对目标帧进行了更新)，CAT就是对目标帧的分割细化
            return out_new,center #[b,c,h,w]

        