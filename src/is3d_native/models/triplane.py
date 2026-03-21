from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .ops import attention_with_fallback


@dataclass
class TriPlaneConfig:
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    plane_channels: int = 24
    plane_resolution: int = 32
    attn_dropout: float = 0.0
    token_mask_ratio: float = 0.0


class TriPlaneDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, attn_dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.attn_dropout = attn_dropout

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = attention_with_fallback(
            q * self.scale,
            k,
            v,
            dropout_p=self.attn_dropout,
            training=self.training,
            force_fallback=self.training,
        )
        out = out.permute(0, 2, 1, 3).reshape(bsz, seq_len, dim)
        return self.proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TriPlaneDecoder(nn.Module):
    def __init__(self, cfg: TriPlaneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.blocks = nn.ModuleList(
            [
                TriPlaneDecoderBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    attn_dropout=cfg.attn_dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        out_dim = 3 * cfg.plane_channels * cfg.plane_resolution * cfg.plane_resolution
        self.head = nn.Linear(cfg.embed_dim, out_dim)

    def _apply_token_masking(
        self,
        tokens: torch.Tensor,
        ratio: float,
        force_masking: bool = False,
    ) -> torch.Tensor:
        if ratio <= 0.0 or (not self.training and not force_masking):
            return tokens

        mask = torch.rand(tokens.shape[0], tokens.shape[1], device=tokens.device) < ratio
        masked = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        return masked

    def forward(
        self,
        tokens: torch.Tensor,
        use_checkpoint: bool = True,
        token_mask_ratio: float | None = None,
        force_masking: bool = False,
    ) -> torch.Tensor:
        ratio = self.cfg.token_mask_ratio if token_mask_ratio is None else token_mask_ratio
        x = self._apply_token_masking(tokens, ratio, force_masking=force_masking)

        for block in self.blocks:
            if use_checkpoint and self.training:
                x = checkpoint(block, x.requires_grad_(True), use_reentrant=True)
            else:
                x = block(x)

        pooled = x.mean(dim=1)
        planes = self.head(pooled)
        return planes.view(
            planes.shape[0],
            3,
            self.cfg.plane_channels,
            self.cfg.plane_resolution,
            self.cfg.plane_resolution,
        )
