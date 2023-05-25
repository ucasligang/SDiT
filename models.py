# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torchvision
from diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
import torch.nn.functional as F

import torch.nn as nn


class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.conv.weight.data.zero_()
        if bias:
            self.conv.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


# By Gang Li.
class Attention(Attention):
    #def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., r=4, dropout_p=0.1, scale=1.0):
        super().__init__(dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # Initilize gamma to 1.0
        self.gamma1 = nn.Parameter(torch.ones(dim * 3))
        self.gamma2 = nn.Parameter(torch.ones(dim))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Apply LoRA
        # self.r = r
        # self.lora_down = nn.Linear(dim, r, bias=False)
        # self.dropout = nn.Dropout(dropout_p)
        # self.lora_up = nn.Linear(r, dim, bias=False)
        # self.selector = nn.Identity()
        #
        # nn.init.normal_(self.lora_down.weight, std=1 / r)
        # nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        B, N, C = x.shape    # 16, 256, 1152
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, C/num_heads]
        # Apply gamma
        # qkv = (self.gamma1 * self.qkv(x)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = (self.gamma1 * self.qkv(x)).reshape(B, N, 3, C)
        # qkv = self.qkv(x).reshape(B, N, 3, C)

        #q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q, k, v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)
        #q = q + self.dropout(self.lora_up(self.selector(self.lora_down(q))))
        #v = v + self.dropout(self.lora_up(self.selector(self.lora_down(v))))
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # Apply gamma
        x = self.gamma2 * self.proj(x)
        x = self.proj_drop(x)
        return x



def modulate(x, shift, scale):
    return x * (1+scale)+shift
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class Preprocess_input(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, patch_size, dropout_prob):
        super().__init__()
        # self.embedding_table = nn.Embedding(num_classes+1+use_cfg_embedding, hidden_size)  # Gang Li.
        #self.embdding_seg = nn.Conv2d(num_classes+1, hidden_size, kernel_size=patch_size*8, stride=patch_size*8)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.dropout_prob = dropout_prob

    # def token_drop(self, labels, force_drop_ids=None):
    #     """
    #     Drops labels to enable classifier-free guidance.
    #     """
    #     if force_drop_ids is None:
    #         drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
    #     else:
    #         drop_ids = force_drop_ids == 1
    #     drop_ids = drop_ids.unsqueeze(-1).unsqueeze(-1).repeat(labels[0].unsqueeze(0).shape)
    #     labels = torch.where(drop_ids, torch.tensor(self.num_classes, dtype=labels.dtype).to(labels), labels)
    #     return labels

    def forward(self, labels, train, force_drop_ids=None):
        # use_dropout = self.dropout_prob > 0
        # if (train and use_dropout) or (force_drop_ids is not None):
        #     labels = self.token_drop(labels, force_drop_ids)
        labels = labels.unsqueeze(1)
        bs, _, h, w = labels.shape  # [2,256,256]
        sample_mask = (labels!=-1)

        input_label = torch.FloatTensor(bs, self.num_classes+1, h, w).zero_().to(labels)
        if not train:
            null = torch.zeros_like(labels)
            labels = torch.where(labels==-1, null, labels)  # make it scatter correctly, then use sample_mask to recover the null label.

        input_semantics = input_label.scatter_(1, labels.long(), 1.0).float()
        # embeddings = self.embedding_table(labels.long())
        if not train:
            input_semantics = input_semantics * sample_mask

        if self.dropout_prob > 0.0 and train:
            mask = (torch.rand([input_semantics.shape[0], 1, 1, 1]) > self.dropout_prob).float()
            input_semantics = input_semantics * mask.to(input_semantics)

        return input_semantics

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, patch_size):
        super().__init__()
        # self.embedding_table = nn.Embedding(num_classes+1+use_cfg_embedding, hidden_size)  # Gang Li.
        #self.embdding_seg = nn.Conv2d(num_classes+1, hidden_size, kernel_size=patch_size*8, stride=patch_size*8)
        self.patch_size = patch_size
        self.embdding_seg = nn.Conv2d(num_classes+1, hidden_size, kernel_size=3, padding=1)

    # def token_drop(self, labels, force_drop_ids=None):
    #     """
    #     Drops labels to enable classifier-free guidance.
    #     """
    #     if force_drop_ids is None:
    #         drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
    #     else:
    #         drop_ids = force_drop_ids == 1
    #     drop_ids = drop_ids.unsqueeze(-1).unsqueeze(-1).repeat(labels[0].unsqueeze(0).shape)
    #     labels = torch.where(drop_ids, torch.tensor(self.num_classes, dtype=labels.dtype).to(labels), labels)
    #     return labels

    def forward(self, input_semantics):
        input_semantics = F.interpolate(input_semantics, size=[32, 32], mode='nearest')
        embeddings = self.embdding_seg(input_semantics)
        return embeddings


class SPADEGroupNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, eps = 1e-5):
        super().__init__()

        self.norm = nn.GroupNorm(32, norm_nc, affine=False) # 32/16
        self.norm_nc = norm_nc
        self.label_nc = label_nc
        self.eps = eps
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        return x * (1 + gamma) + beta


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, num_classes, patch_size, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Initilize gamma to 1.0
        self.gamma1 = nn.Parameter(torch.ones(hidden_size))
        self.gamma2 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, c):

        # c = t.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        #x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        #x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + self.gamma1 * gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.gamma2 * gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class OutNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, eps = 1e-5):
        super().__init__()

        # self.norm = nn.GroupNorm(32, norm_nc, affine=False) # 32/16
        self.norm = nn.LayerNorm(norm_nc, elementwise_affine=True, eps=1e-6)

        self.eps = eps
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            #nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            ZeroConv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_gamma = ZeroConv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        # self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = ZeroConv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations N,C,H,W
        x = x.permute(0,2,3,1)  # N,H,W,C
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        return x * (1 + gamma) + beta


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, num_classes):
        super().__init__()

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)   # False update in 20230506
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, patch_size)

    def forward(self, x, c):  # x:256 c:1024
        # y = self.y_embedder(y).permute(0, 2, 3, 1)
        # t = t.unsqueeze(1).unsqueeze(1).repeat(1, y.shape[1], y.shape[2], 1)
        # #print(t.shape)
        # c = y+t
        # c = c.reshape(c.shape[0], -1, c.shape[-1])
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1) # [2,256,1152]

        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # [2,256,32]
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        #self.label_resize1 = torchvision.transforms.Resize([input_size//2, input_size//2], torchvision.transforms.InterpolationMode.NEAREST)
        #self.label_resize2 = torchvision.transforms.Resize([input_size, input_size], torchvision.transforms.InterpolationMode.NEAREST)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.preprocess_input = Preprocess_input(num_classes, hidden_size, patch_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, num_classes, patch_size, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, patch_size)
        # self.out_norm = OutNorm(self.out_channels, hidden_size)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, num_classes)
        self.initialize_weights() # Gang Li.

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)

        y = self.preprocess_input(y, self.training) # self.y_embedder(y, self.training).permute(0, 2, 3, 1)  # [2, 1152, 256, 256]-->[2, 256, 256, 1152]

        y = self.y_embedder(y).permute(0, 2, 3, 1)
        t = t.unsqueeze(1).unsqueeze(1).repeat(1, y.shape[1], y.shape[2], 1)
        c = y+t
        c = c.reshape(c.shape[0], -1, c.shape[-1])

        # c1 = t + y                                # (N,H/ patch_size,W/ patch_size,D)
        # c1 = c1.reshape(x.shape[0], -1, x.shape[-1])  # (N,T,D)

        #t2 = t.unsqueeze(1).unsqueeze(1).repeat(1, y2.shape[1], y2.shape[2], 1)
        #c2 = t2+y2  # (N,H/ patch_size,W/ patch_size,D)
        # c2 = c2.reshape(x.shape[0], -1, x.shape[-1])  # (N,T,D)


        for block in self.blocks:
            #x = block(x, t, y)                      # (N, T, D)
            x = block(x, c)

        # x = self.final_layer(x, t, y)                # (N, T, patch_size ** 2 * out_channels) y:[2, 151, 256, 256]
        x = self.final_layer(x, c)
        x = self.unpatchify(x)          # (N, out_channels, H, W)  # x:[2,8,32,32] c2:[2, 32*32, D]

        # c2 = c2.permute(0, 3, 1, 2)
        # x = self.out_norm(x, c2)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        #half_eps = cond_eps
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)
    y = torch.tensor(class_labels, device=device)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0).to(device)

    model = DiT_XL_2().cuda()
    x = torch.rand([16, 4, 32, 32], device=device)
    t = torch.tensor([250]*16, device=device)
    y = model(x,t,y)
    print(y.shape)

