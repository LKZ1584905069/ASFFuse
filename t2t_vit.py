import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        # x = x.cuda()

        # out = self.conv2d(x)
        # out = self.bn(out)
        out = self.reflection_pad(x)

        out = self.conv2d(out)

        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out

# transformer中的MLP
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

# transformer 的注意力机制
# # dim=128, depth=4,
# # num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0.,
# # attn_drop_rate=0.,drop_path_rate=0., norm_layer=nn.LayerNorm
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

        # @ 矩阵乘法
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# dim=128, depth=4,
# num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0.,
# attn_drop_rate=0.,drop_path_rate=0., norm_layer=nn.LayerNorm
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # 注意力机制
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # nn.Identity()输入是啥，直接给输出，不做任何的改变
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# 将向量转成 pxp的像素
class C_DePatch(nn.Module):
    def __init__(self, channel=3, embed_dim=128, patch_size=16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, patch_size**2),
        )
   
    def forward(self, x, ori):
        b, c, h, w = ori
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)
        x = rearrange(x, '(b h w) c (p1 p2) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        return x

#class C_DePatch(nn.Module):
#    def __init__(self, channel=3, embed_dim=128, patch_size=16):
#        self.patch_size = patch_size
#        super().__init__()
#        self.projection = nn.Sequential(
#            nn.Linear(embed_dim, patch_size**2),
#        )
#        self.f = nn.Linear(channel, 1)
#
#    def forward(self, x, ori):
#        b, c, h, w = ori
#        h_ = h // self.patch_size
#        w_ = w // self.patch_size
#        x = self.projection(x)
#        x = rearrange(x, '(b h w) c (p1 p2) -> (b h w) (p1 p2) c', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
#        x = self.f(x)
#        x = rearrange(x, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
#        return x

class S_DePatch(nn.Module):
    def __init__(self, channel=16, embed_dim=128, patch_size=16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, patch_size**2),
        )

    def forward(self, x, ori):
        b, c, h, w = ori
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        return x


# 对应两个通道的 MLP 部分
# embed_dim=128, depth=4,
# num_heads=4, mlp_ratio=2., patch_size=16,qkv_bias=False, qk_scale=None, drop_rate=0.,
# attn_drop_rate=0.,drop_path_rate=0., norm_layer=nn.LayerNorm
class encoder(nn.Module):
    def __init__(self, embed_dim=256, depth=4,
                 num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        # self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pos_drop = nn.Dropout(p=drop_rate)

        # torch.linspace(star,end,step)返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
        # transformer中的 N 个注意力机制的 dpr
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # transformer中的 N 个注意力机制
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
            # 一个的输出是下一个的输入
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x


# Channel(size=32, embed_dim=128, patch_size=16, channel=64)
# embed_dim 是每个块拉成的向量长度
class Channel(nn.Module):
    def __init__(self, size=224,embed_dim=128, depth=4, channel=16,
                 num_heads=4, mlp_ratio=2., patch_size=16,qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        # self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # 维度转换
        self.embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size, p2=patch_size),
            # 把每个块拉成向量 patch_size**2 一个块的像素值， embed_dim 目标向量长度，也是图中对应的MLP
            nn.Linear(patch_size**2, embed_dim),
        )


        # 随机丢弃的神经元
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer
        self.en = encoder(embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                 drop_path_rate, norm_layer)

        # 将向量转成 pxp的像素
        # rearrange(x, '(b h w) c (p1 p2) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        self.depatch = C_DePatch(channel=channel, embed_dim=embed_dim, patch_size=patch_size)

    def forward(self, x):
        ori = x.shape # ori  = torch.Size([16, 100, 32, 32])
        x2_t = self.embedding(x)
        x2_t = self.pos_drop(x2_t)
        x2_t = self.en(x2_t)
        out = self.depatch(x2_t, ori)
        return out

#  embed_dim=1024*2, patch_size=4, channel=in_channel/2
class Spatial(nn.Module):
    def __init__(self, size=256, embed_dim=128, depth=4, channel=16,
                 num_heads=4, mlp_ratio=2., patch_size=16, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        # self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size ** 2 * channel, embed_dim),
        )
        # self.embedding = nn.Conv2d(in_chans, embed_dim*in_chans, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        # self.linear = nn.Linear(embed_dim, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.en = encoder(embed_dim, depth,
                          num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                          drop_path_rate, norm_layer)

        self.depatch = S_DePatch(channel=channel, embed_dim=embed_dim, patch_size=patch_size)

    def forward(self, x):
        ori = x.shape
        x2_t = self.embedding(x)
        x2_t = self.pos_drop(x2_t)
        x2_t = self.en(x2_t)
        out = self.depatch(x2_t, ori)
        return out



