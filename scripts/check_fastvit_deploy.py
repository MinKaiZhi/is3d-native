from __future__ import annotations

import argparse

import torch

from is3d_native.models.fastvit import FastViTBackbone, FastViTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FastViT switch_to_deploy output drift.")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)

    cfg = FastViTConfig(
        in_channels=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=2.0,
        patch_size=8,
        attn_dropout=0.0,
    )

    model = FastViTBackbone(cfg).eval()
    images = torch.randn(args.batch_size, 3, args.image_size, args.image_size)

    with torch.no_grad():
        out_before = model(images, use_checkpoint=False)
        model.switch_to_deploy()
        out_after = model(images, use_checkpoint=False)

    max_abs_err = (out_before - out_after).abs().max().item()
    print(f"max_abs_err={max_abs_err:.6e}")


if __name__ == "__main__":
    main()
