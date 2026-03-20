from __future__ import annotations

import torch

from is3d_native.models.triplane import TriPlaneConfig, TriPlaneDecoder


def test_triplane_token_masking_full_ratio() -> None:
    cfg = TriPlaneConfig(
        embed_dim=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        plane_channels=4,
        plane_resolution=8,
        token_mask_ratio=1.0,
    )
    model = TriPlaneDecoder(cfg)
    model.train()

    tokens = torch.randn(2, 10, 16)
    masked = model._apply_token_masking(tokens, ratio=1.0)

    target = model.mask_token.expand_as(tokens)
    assert torch.allclose(masked, target)


def test_triplane_forward_accepts_mask_ratio_override() -> None:
    cfg = TriPlaneConfig(
        embed_dim=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        plane_channels=4,
        plane_resolution=8,
        token_mask_ratio=0.0,
    )
    model = TriPlaneDecoder(cfg)
    model.train()

    tokens = torch.randn(2, 10, 16)
    out = model(tokens, use_checkpoint=False, token_mask_ratio=0.25)
    assert out.shape == (2, 3, 4, 8, 8)
