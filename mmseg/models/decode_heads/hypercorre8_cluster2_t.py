# from CLUSTERFORMER: Clustering As A Universal Visual Learner
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
from .fdsf import FDSF,PagFM
from flash_attn import flash_attn_func 




class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
    x1 = F.normalize(x1,dim=-1)
    x2 = F.normalize(x2,dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Clustering(nn.Module):
    def __init__(self, dim, out_dim, center_w=2, center_h=2, window_w=2, window_h=2, heads=4, head_dim=24, return_center=False, num_clustering=1):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.conv1 = nn.Conv2d(dim, heads*head_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(heads*head_dim, out_dim, kernel_size=1)
        self.conv_c = nn.Conv2d(head_dim, head_dim, kernel_size=1)
        self.conv_v = nn.Conv2d(dim, heads*head_dim, kernel_size=1)
        self.conv_f = nn.Conv2d(dim, heads*head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((center_w,center_h))
        self.window_w = int(window_w)
        self.window_h = int(window_h)
        self.return_center = return_center
        self.softmax = nn.Softmax(dim=-2)
        self.num_clustering = num_clustering

    def forward(self, x,t,img_metas=None): #[bt,c,w,h] #t第一层为4，后面为1
        _,_,W,H = x.shape
        value = self.conv_v(x) 

        # frame_x = rearrange(x, "(b t) c w h -> b t c h w",t=t)
        # tg_x = frame_x[:,-1,:,:,:]
        # feature = self.conv_f(tg_x)

        feature = self.conv_f(x)

        x = self.conv1(x) # create_center all_t,这里需要高级还是低级特征合适？多尺度聚类？，如果高级可以扩大均值池化的窗口(EFC中采用的4x4) 可以在前面不进行拼接，直接进行多层次的clustering,层次之间的center不同。

        # multi-head
        b, c, w, h = x.shape
        x = x.reshape(b*self.heads, int(c/self.heads), w, h)
        value = value.reshape(b*self.heads, int(c/self.heads), w, h)
        feature = feature.reshape(b*self.heads, int(c/self.heads), w, h)
        # print("out",x.shape,feature.shape,value.shape)
        # window token
        pad_l = pad_r = pad_t = pad_b = 0
        if self.window_w>1 and self.window_h>1: # 不重叠windows
            b, c, w, h = x.shape
            # # pad
            pad_l = pad_t = 0
            pad_b = (self.window_w - w % self.window_w) % self.window_w
            pad_r = (self.window_h - h % self.window_h) % self.window_h
            # 这里的w,h是相反的
            x = F.pad(x,(pad_l,pad_r,pad_t,pad_b))
            value = F.pad(value,(pad_l,pad_r,pad_t,pad_b))
            feature = F.pad(feature,(pad_l,pad_r,pad_t,pad_b))
            # # print("x",x.shape,self.window_h,self.window_w)
            b, c, w, h = x.shape
            x = x.reshape(b*self.window_w*self.window_h, c, int(w/self.window_w), int(h/self.window_h))
            value = value.reshape(b*self.window_w*self.window_h, c, int(w/self.window_w), int(h/self.window_h))
            feature = feature.reshape(b*self.window_w*self.window_h, c, int(w/self.window_w), int(h/self.window_h))

        b, c, w, h = x.shape #[bt*head*window, c, w, h]
        value = value.reshape(b, w*h, c) # (bt*head*window,n,c)

        # centers
        centers = self.centers_proposal(x) #Avg Q #这里能保持聚类中心的特征维度与目标维度一致
        b, c, c_w, c_h = centers.shape
        centers_feature = self.centers_proposal(feature) #Avg K (bt*head*window,c,cw,ch)
        # print("centers_feature",centers_feature.shape)
        centers_feature = centers_feature.reshape(-1,t,self.heads,self.window_w*self.window_h, c, c_w, c_h)
        centers_feature = centers_feature.permute(0,2,3,4,1,5,6).reshape(-1,c,t*c_w*c_h) # (b*head*window,c,t*n_avg)
        # print("after_centers_feature",centers_feature.shape)

        # processing before flash attention
        centers = centers.reshape(int(b/self.heads), c_w*c_h, self.heads, c).type(torch.half) #(bt*window,n_avg,head,c) 
        value = value.reshape(int(b/self.heads), w*h, self.heads, c).type(torch.half) #(bt*window,n,head,c)
        feature = feature.reshape(int(b/self.heads), w*h, self.heads, c).type(torch.half) #(bt*window,n,head,c)

        # 这里得到的聚类结果考虑用backbone得到的高置信度类别进行损失计算，高置信度在deep认为容易分割，但是这对聚类更新是平等的。
        # 这里对时间t加上置信度理论！！！将不同的时间理解为多视图问题,需要考虑差异，只有高置信度或者是前后都出现的物体才可以更新(后续)
        for _ in range(self.num_clustering):    # iterative clustering and updating centers
            centers = flash_attn_func(centers, value, feature) # all_t (bt*window,n,head,c) # 这里可以测试近高亲合并
        # 前面feteature参与了centers的产生过程，不能去掉mem
        

        # print("centers",centers.shape)
        frame_feature = feature.reshape(-1, t, self.window_w*self.window_h,w,h, self.heads, c) # (bt*window,n,head,c)
        # print("frame_feature",frame_feature.shape)
        tg_feature = frame_feature[:,-1,:,:,:] # (b,window,w,h,head,c)
        tg_feature = tg_feature.permute(0,4,1,5,2,3).reshape(-1, c, w, h) # (b*head*window,c,w,h)
        # print("tg_feature",tg_feature.shape)

        # processing after flash attention
        # tg_frame
        b_tg,_,_,_ = tg_feature.shape

        centers = centers.reshape(b, c, c_w, c_h).type(torch.float) #(bt*head*window,c,cw,ch) 
        value = value.reshape(b, w*h, c).type(torch.float) #(bt*head*window,n,c)
        tg_feature = tg_feature.reshape(b_tg, w*h, c).type(torch.float) # (b*head*window,n,c)
        #x [bt*head*window, c, w, h]
        frame_x = x.reshape(-1,t,self.heads,self.window_w*self.window_h, c, w, h)
        tg_x = frame_x[:,-1,:,:,:,:,:].reshape(-1, w*h, c) # (b*head*window, c, w, h) 
        # print("centers,value,tg_feature,tg_x",centers.shape,value.shape,tg_feature.shape,tg_x.shape)

        centers = centers.reshape(-1,t,self.heads,self.window_w*self.window_h, c, c_w, c_h)
        centers = centers.permute(0,2,3,4,1,5,6).reshape(b_tg,c,t,c_w,c_h) # (b*head*window,c,t,cw,ch)
        
        # similarity sim(centers(b*head*window,t*n_avg,c) , x(b*head*window,c,n)) -> (b*head*window,t*n_avg,n) 这里的x应该是tg_x，表示对目标帧的分配,这里可以产生离线和在线的说法，
        # 可以一次性分割多个帧,应为cluster是共享的,现在测试tg_x的情况
        similarity = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(centers.reshape(b_tg,c,-1).permute(0,2,1), tg_x.reshape(b_tg,c,-1).permute(0,2,1)))
        
        # assign each point to one center
        _, max_idx = similarity.max(dim=1, keepdim=True)
        mask = torch.zeros_like(similarity)
        mask.scatter_(1, max_idx, 1.)
        similarity= similarity*mask # 只选取相似度最大的cluster, feture应该是target frame
        # centers_feature: # (b*head*window,c,t*cw*ch)
        # add1: (b*head*window,t*n_avg,c)
        # eq.6 in the paper (b*head*window,1,n,c) * (b*head*window,t*n_avg,n,1) -> (b*head*window,t*n_avg,n,c) .sum -> (b*window,t*n_avg,c) 对cluster加权
        out = (( tg_feature.unsqueeze(dim=1)*similarity.unsqueeze(dim=-1) ).sum(dim=2) + centers_feature.transpose(-1,-2))/ (mask.sum(dim=-1,keepdim=True)+ 1.0) 

        if self.return_center:
            out = out.reshape(b, c, c_w, c_h)
            return out
        else:
            out = (out.unsqueeze(dim=2)*similarity.unsqueeze(dim=-1)).sum(dim=1) # (b*head*window,t*n_avg,1，c) * (b*head*window,t*n_avg,n,1) -> (b*window,t*n_avg,n,c) .sum -> (b*head*window,n,c)
            out = out.reshape(b_tg, c, w, h)

        # recover feature maps
        if self.window_w>1 and self.window_h>1:
            out = out.reshape(int(out.shape[0]/self.window_w/self.window_h), out.shape[1], out.shape[2]*self.window_w, out.shape[3]*self.window_h)
        
        out = out.reshape(int(out.shape[0]/self.heads), out.shape[1]*self.heads, out.shape[2], out.shape[3]) #(b,c,w,h)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :W, :H].contiguous()
        out = self.conv2(out)
        return out




# class Mlp(nn.Module):
#     """ Multilayer perceptron."""

#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,need_dw=True):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.need_dw = need_dw
#         if need_dw:
#             self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x, H, W):
#         x = self.fc1(x)
#         if self.need_dw:
#             x = self.dwconv(x, H, W)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.,need_dw=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.need_dw = need_dw

        if need_dw:
            self.dwconv = DWConv(hidden_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        # print("mlp",x.shape)
        x = self.fc1(x)
        if self.need_dw:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# depth-wise conv
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
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

class ClusterBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 center_w=2, center_h=2, window_w=2, window_h=2, heads=4, head_dim=24, return_center=False,num_clustering=1):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Clustering(dim=dim, out_dim=dim, center_w=center_w, center_h=center_h, window_w=window_w, window_h=window_h, heads=heads, head_dim=head_dim, return_center=return_center,num_clustering=num_clustering)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        self.return_center = return_center
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x,H,W,mem=None,t=1,img_metas=None): #[bt,n,c]
        if mem is not None:
            mem = rearrange(mem, "(B T) (H W) C -> B T C H W ", T = t,H=H, W=W).contiguous()
            tg_x = rearrange(x, "B (H W) C -> B C H W ", H=H, W=W).contiguous()
            x = torch.cat([mem,tg_x.unsqueeze(1)],dim=1)
            x = x.flatten(0,1) #[bt,C,H,W]
            t = 4
        else:
            x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W).contiguous()
            tg_x = x
            t = 1

        if self.use_layer_scale:
            out1 = self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x),t,img_metas=img_metas))
            # print("out1",out1.shape,tg_x.shape)
            x = tg_x + out1
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x),H,W))
        else:
            x = tg_x + self.drop_path(self.token_mixer(self.norm1(x),t,img_metas=img_metas))
            x = x + self.drop_path(self.mlp(self.norm2(x),H,W))
        
        x = rearrange(x, "B C H W -> B (H W) C").contiguous()
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio = 4.,
                 act_layer = nn.GELU, norm_layer = GroupNorm,
                 drop_rate = .0, drop_path_rate = 0.,
                 use_layer_scale=True, layer_scale_init_value = 1e-5,
                 center_w = 5, center_h = 5, window_w = 5, window_h = 5, heads = 4, head_dim = 24, return_center = False, num_clustering=1):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(ClusterBlock(
            dim, mlp_ratio = mlp_ratio,
            act_layer = act_layer, norm_layer = norm_layer,
            drop = drop_rate, drop_path = block_dpr,
            use_layer_scale = use_layer_scale,
            layer_scale_init_value = layer_scale_init_value,
            center_w = center_w, center_h = center_h, window_w = window_w, window_h = window_h,
            heads=heads, head_dim=head_dim, return_center=return_center, num_clustering= num_clustering
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class Cluster_layer(nn.Module):
    def __init__(self, dim, num_heads,  num_clusters,mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.norm1 = norm_layer(dim)
        self.clustering = Cluster_Block(dim,num_clusters,mlp_ratio=mlp_ratio)
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

    def forward(self,x,H,W,z=None,mem=None,t=3):
        '''
            x:[b,n,c]
            mem:[bt,n,c]
            z:[b,num_clusters,tn] # 选择迭代更新
        '''
        res = x
        if mem is not None:
            x = torch.cat([mem.view(-1,t,mem.shape[-2],mem.shape[-1]),x.unsqueeze(1)],dim=1) #[b,t,n,c]
            x = x.flatten(0,1) #[bt,n,c]
            # print("x",x.shape) #x torch.Size([2, 14400, 256])
        # clustering
        cluster_x_z = self.clustering(x,H,W) # [bt,num_clusters,n]
        cluster_x_z = rearrange(cluster_x_z,"(b t) c n -> b t c n",b=res.shape[0])
        cluster_x_z = cluster_x_z.permute(0,2,1,3).contiguous().flatten(2) # [b,num_clusters,tn]
        if z is not None:
        # 提供一个融合的模块，来完成cluster的更新，,进行token_selection然后重组
          # 先select 再求平均(k_means)
            z = rearrange(z,'b c n -> b n c')
            cluster_x_z = rearrange(cluster_x_z,'b c n -> b n c')
            # select:
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
            cluster_x_z = rearrange(cluster_x_z,"b n c -> b c n")

        cluster_x = cluster_x_z.softmax(dim=-1) # [b,num_clusters,tn]
        x = rearrange(x,'(b t) n c -> b (t n) c',b=res.shape[0])
        
        C_in = cluster_x @ x #vison_token [b,num_clusters,c]
        C_in = self.norm1(C_in)
        src = rearrange(res,"b n c -> n b c")
        mem = rearrange(C_in, "b n c -> n b c")
        out,attn = F.multi_head_attention_forward(
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
            average_attn_weights=False,
        )
        out = rearrange(out,"n b c -> b n c")
        out = self.proj_drop(out)
        
        out = res + self.norm1(out)
        out = res + self.norm1(self.mlp(out,H,W))

        return out,cluster_x_z
    
    
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


# CATBlock： ipsa: 建模静态块内， cpsa:建模全局的动态行为(这里应该有交互吗？） time_layer:建模时间交互， out：linear融合多个时间结果，需要考虑空间？

# 多尺度的交互： 逐渐上采样的过程中，可以有avg(x) - x来增强细节和边界(以及证明)

# 将聚类贯穿到底
class hypercorre_topk2(nn.Module):
    """ top-k2: same selections for each reference image so that attention decoder can be used

    Args:
    num_feats: number of features being used

    """

    def __init__(self,dim=[64, 128, 320, 512], num_layers=1, t=3, time_decoder_layer=3,embedding_dim=256,num_classes = 124,cross_method='CAT',ratio_fusio = False,num_clusters=150):
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
        self.t = 1

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
        
        self.cluster_blocks = basic_blocks(
                                    dim = embedding_dim,
                                    index = 0,
                                    layers=[num_layers],
                                    mlp_ratio=4.,
                                    norm_layer=GroupNorm,
                                    center_h=12,
                                    center_w=12,
                                    window_h=6,
                                    window_w=6,
                                    heads=8,
                                    head_dim=32
                                )


        self.class_token = Class_Token_Seg3(dim=embedding_dim, num_heads=1,num_classes=num_classes,T=self.t)
        self.trans = TransformerClassToken3(dim=embedding_dim, depth=1, num_heads=4,  num_classes =  num_classes,
                                            trans_with_mlp=True, att_type="SelfAttentionWithTime")
        
    def forward(self, query_frame, supp_frame,img_metas = None,t=None):
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

        B,num_clips,C,H,W = memory.shape
        z = None

        src = rearrange(src,'b t c h w -> (b t) c (h w)')
        memory = rearrange(memory,'b t c h w -> (b t) c (h w)')
        src = src.permute(0,2,1)
        memory = memory.permute(0,2,1)
        centers = None
        
        for idx,blk in enumerate(self.cluster_blocks):
            if idx == 0:
                x = blk(src, H=H, W=W, mem = memory, t=T_pre,img_metas=img_metas)
            else:
                x = blk(x, H=H, W=W,t=T_tg,img_metas=img_metas) #[b,n,c] 这里在前几层添加mem似乎有效
    
        x = rearrange(x, '(b t) (H W) c -> b t c H W', b=B, t=T_tg,H=H,W=W)
        # 同Focal,只有tg得到更新
        # 可学习的原型 (B,num_clips,C,H,W)
        out_cls_mid, cls_tokens =  self.class_token(x)
        out_new = self.trans(x, cls_tokens, out_cls_mid) #bxtxcxhxw
        out_new=(torch.chunk(out_new, T_tg, dim=1))
        out_new=[ii.squeeze(1) for ii in out_new] # focal情况下就是对所有帧的分割细化(还对目标帧进行了更新)，CAT就是对目标帧的分割细化
        return out_new,out_cls_mid.squeeze(1),centers #[b,c,h,w]
