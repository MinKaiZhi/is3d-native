from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .fastvit import FastViTBackbone, FastViTConfig
from .triplane import TriPlaneConfig, TriPlaneDecoder


@dataclass
class IS3DModelConfig:
    fastvit: FastViTConfig = field(default_factory=FastViTConfig)
    triplane: TriPlaneConfig = field(default_factory=TriPlaneConfig)


class IS3DModel(nn.Module):
    def __init__(self, cfg: IS3DModelConfig) -> None:
        super().__init__()
        if cfg.fastvit.embed_dim != cfg.triplane.embed_dim:
            raise ValueError("FastViT embed_dim must equal TriPlane embed_dim")
        self.backbone = FastViTBackbone(cfg.fastvit)
        self.decoder = TriPlaneDecoder(cfg.triplane)

    def forward(
        self,
        images: torch.Tensor,
        use_checkpoint: bool = True,
        token_mask_ratio: float | None = None,
    ) -> torch.Tensor:
        tokens = self.backbone(images, use_checkpoint=use_checkpoint)
        return self.decoder(
            tokens,
            use_checkpoint=use_checkpoint,
            token_mask_ratio=token_mask_ratio,
        )

    def switch_to_deploy(self) -> None:
        self.backbone.switch_to_deploy()
