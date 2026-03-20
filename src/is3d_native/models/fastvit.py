from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .ops import attention_with_fallback


@dataclass
class FastViTConfig:
    in_channels: int = 3
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    patch_size: int = 8
    attn_dropout: float = 0.0


class ReparamLinear(nn.Module):
    """
    Multi-branch linear layer merged to a single branch for deploy.

    Merge path uses float64 accumulation, then casts back to float32 for runtime.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_identity = in_features == out_features

        self.branch_main = nn.Linear(in_features, out_features, bias=bias)
        self.branch_aux1 = nn.Linear(in_features, out_features, bias=bias)
        self.branch_aux2 = nn.Linear(in_features, out_features, bias=bias)
        self.reparam: nn.Linear | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam is not None:
            return self.reparam(x)

        out = self.branch_main(x)
        out = out + self.branch_aux1(x)
        out = out + self.branch_aux2(x)
        if self.has_identity:
            out = out + x
        return out

    def _merge_branches(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.branch_main.weight.device
        merged_weight = torch.zeros(
            self.out_features,
            self.in_features,
            device=device,
            dtype=torch.float64,
        )
        merged_bias = torch.zeros(self.out_features, device=device, dtype=torch.float64)

        for branch in (self.branch_main, self.branch_aux1, self.branch_aux2):
            merged_weight += branch.weight.detach().to(torch.float64)
            if branch.bias is not None:
                merged_bias += branch.bias.detach().to(torch.float64)

        if self.has_identity:
            eye = torch.eye(self.out_features, device=device, dtype=torch.float64)
            merged_weight += eye

        return merged_weight.to(torch.float32), merged_bias.to(torch.float32)

    def switch_to_deploy(self) -> None:
        if self.reparam is not None:
            return

        merged_weight, merged_bias = self._merge_branches()
        layer = nn.Linear(self.in_features, self.out_features, bias=True).to(merged_weight.device)

        with torch.no_grad():
            layer.weight.copy_(merged_weight)
            layer.bias.copy_(merged_bias)

        self.reparam = layer
        del self.branch_main
        del self.branch_aux1
        del self.branch_aux2


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = attention_with_fallback(
            q * self.scale,
            k,
            v,
            dropout_p=self.attn_dropout,
            training=self.training,
            force_fallback=self.training,
        )
        out = attn.permute(0, 2, 1, 3).reshape(bsz, seq_len, dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = ReparamLinear(dim, hidden)
        self.fc2 = ReparamLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    def switch_to_deploy(self) -> None:
        self.fc1.switch_to_deploy()
        self.fc2.switch_to_deploy()


class FastViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, attn_dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def switch_to_deploy(self) -> None:
        self.mlp.switch_to_deploy()


class FastViTBackbone(nn.Module):
    def __init__(self, cfg: FastViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            [
                FastViTBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    attn_dropout=cfg.attn_dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, images: torch.Tensor, use_checkpoint: bool = True) -> torch.Tensor:
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)

        for block in self.blocks:
            if use_checkpoint and self.training:
                x = checkpoint(block, x.requires_grad_(True), use_reentrant=True)
            else:
                x = block(x)

        return self.norm(x)

    def switch_to_deploy(self) -> None:
        for block in self.blocks:
            block.switch_to_deploy()



