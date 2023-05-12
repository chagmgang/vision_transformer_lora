import torch
import torch.nn as nn
from .base_model import VisionTransformer, Block
from functools import partial

class LoRAAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        rank=2,
    ):
        super(LoRAAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.q_lora_a = nn.Linear(head_dim, rank, bias=False)
        self.q_lora_b = nn.Linear(rank, head_dim, bias=False)
        self.v_lora_a = nn.Linear(head_dim, rank, bias=False)
        self.v_lora_b = nn.Linear(rank, head_dim, bias=False)

        nn.init.normal_(self.q_lora_a.weight, std=0.02)
        self.q_lora_b.weight.data.zero_()
        nn.init.normal_(self.v_lora_a.weight, std=0.02)
        self.v_lora_b.weight.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_delta = self.adapter_forward(
            q,
            self.q_lora_a.weight,
            self.q_lora_b.weight,
        )
        v_delta = self.adapter_forward(
            v,
            self.v_lora_a.weight,
            self.v_lora_b.weight,
        )
        q = q.contiguous() + q_delta
        v = v.contiguous() + v_delta

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

    def adapter_forward(self, x, weight_1, weight_2, g_weight=None):
        result = torch.matmul(x, weight_1.type_as(x).T)
        final = torch.matmul(result, weight_2.type_as(x).T)
        return final

class LoRABlock(Block):

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        rank=2,
    ):
        super(LoRABlock, self).__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer,
        )

        self.attn = LoRAAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            rank=rank,
        )

class LoRAVisionTransformer(VisionTransformer):

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
        num_classes=10,
        rank=2,
    ):
        super(LoRAVisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            with_cp=with_cp,
            num_classes=num_classes)

        self.blocks = nn.ModuleList([
            LoRABlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                rank=rank,
            ) for i in range(depth)])
