# 2023.0513 @Brian

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
from timm.models.layers import DropPath, LayerNorm2d, to_2tuple

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0,
            scale_learnable=True,
            bias_learnable=True,
            mode=None,
            inplace=False
    ):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
    
class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True, use_nchw=True):
        super().__init__()
        self.shape = (dim, 1, 1) if use_nchw else (dim,)
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale.view(self.shape)
    
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)
    
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        stdv = 1. / math.sqrt(d/3)
        self.scale.data.uniform_(-stdv, stdv)

        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    """

    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        y = self.pool(x)
        return y - x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RectifiedLinearAttention(nn.Module):
    """ Rectified Linear Attention
    This repo contain pytorch implementation of 'Sparse Attention with Linear Units'.
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., rmsnorm=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.norm = RMSNorm(inner_dim) if rmsnorm else nn.LayerNorm(inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = F.relu(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out =  self.to_out(self.norm(out))
        return out

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class RLABlock(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RectifiedLinearAttention(dim, rmsnorm=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class MLPMixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self,
            dim,
            seq_len,
            mlp_ratio=(0.5, 4.0),
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x

class ResMLPBlock(nn.Module):
    """ Residual MLP block w/ LayerScale and Affine 'norm'

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    def __init__(
            self,
            dim,
            seq_len,
            mlp_ratio=4,
            mlp_layer=Mlp,
            norm_layer=Affine,
            act_layer=nn.GELU,
            init_values=1e-4,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.linear_tokens = nn.Linear(seq_len, seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        return x

class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(
            self,
            dim,
            token_mixer=Pooling,
            mlp_act=StarReLU,
            mlp_bias=False,
            norm_layer=LayerNorm2d,
            proj_drop=0.,
            drop_path=0.,
            use_nchw=True,
            layer_scale_init_value=None,
            res_scale_init_value=None,
            **kwargs
    ):
        super().__init__()

        ls_layer = partial(Scale, dim=dim, init_value=layer_scale_init_value, use_nchw=use_nchw)
        rs_layer = partial(Scale, dim=dim, init_value=res_scale_init_value, use_nchw=use_nchw)

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, proj_drop=proj_drop, **kwargs)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = ls_layer() if layer_scale_init_value is not None else nn.Identity()
        self.res_scale1 = rs_layer() if res_scale_init_value is not None else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            dim,
            int(4 * dim),
            act_layer=mlp_act,
            # bias=mlp_bias,
            drop=proj_drop,
            # use_conv=use_nchw,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = ls_layer() if layer_scale_init_value is not None else nn.Identity()
        self.res_scale2 = rs_layer() if res_scale_init_value is not None else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x
    

class Transformer(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=27):
        super().__init__()
        drop_path_rate = 0.2
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        # Transformer Block with Attention @Paper
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=h, 
                mlp_hidden_dim=mlp_hidden_dim, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer)
            for i in range(depth)])

        # 2023.0517 Transformer Block with Rectified Linear Attention @Brian
        # self.blocks = nn.ModuleList([
        #     RLABlock(
        #         dim=embed_dim, 
        #         mlp_hidden_dim=mlp_hidden_dim, 
        #         drop=drop_rate, 
        #         drop_path=dpr[i], 
        #         norm_layer=norm_layer)
        #     for i in range(depth)])

        # 2023.0513 MLPMixerBlock @Brian
        # self.blocks = nn.ModuleList([
        #     MLPMixerBlock(
        #         embed_dim,
        #         length,
        #         mlp_ratio=(0.25, 2.0),
        #         mlp_layer=Mlp,
        #         norm_layer=norm_layer,
        #         act_layer=nn.GELU,
        #         drop=0.,
        #         drop_path=0.)
        #     for i in range(depth)])

        # 2023.0517 ResMLPBlock @Brian
        # self.blocks = nn.ModuleList([
        #     ResMLPBlock(
        #         embed_dim,
        #         length,
        #         mlp_ratio=2.0,
        #         mlp_layer=Mlp,
        #         norm_layer=norm_layer,
        #         act_layer=nn.GELU,
        #         drop=0.,
        #         drop_path=0.)
        #     for i in range(depth)])
        
        # 2023.0514 MetaFormerBlock @Brian
        # self.blocks = nn.ModuleList([
        #     MetaFormerBlock(
        #         dim=embed_dim,
        #         token_mixer=Pooling,
        #         mlp_act=StarReLU,
        #         mlp_bias=False,
        #         norm_layer=norm_layer,
        #         proj_drop=0.,
        #         drop_path=dpr[i],
        #         layer_scale_init_value=None,
        #         res_scale_init_value=None,
        #         use_nchw=not issubclass(Pooling, Attention))
        #     for i in range(depth)])

        self.norm = norm_layer(embed_dim)


    def forward(self, x):
        x += self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x