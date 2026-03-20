from __future__ import annotations

import torch

from is3d_native.models.fastvit import FastViTBackbone, FastViTConfig


def test_fastvit_switch_to_deploy_output_close() -> None:
    torch.manual_seed(7)
    cfg = FastViTConfig(
        in_channels=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        patch_size=8,
        attn_dropout=0.0,
    )
    model = FastViTBackbone(cfg).eval()
    images = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        before = model(images, use_checkpoint=False)
        model.switch_to_deploy()
        after = model(images, use_checkpoint=False)

    max_abs_err = (before - after).abs().max().item()
    assert max_abs_err < 1e-5
