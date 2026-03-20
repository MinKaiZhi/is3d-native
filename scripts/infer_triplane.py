from __future__ import annotations

import argparse
from pathlib import Path

import torch

from is3d_native.inference import (
    choose_device,
    chunked_inference,
    load_exported_model,
    preprocess_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run triplane inference with chunk mode.")
    parser.add_argument("--export-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/infer/triplane.pt"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--input-images",
        type=Path,
        nargs="+",
        default=None,
        help="One or more image paths. If omitted, synthetic input is used.",
    )
    parser.add_argument("--synthetic-batch", type=int, default=1)
    return parser.parse_args()


def _load_batch(args: argparse.Namespace, preprocess: dict) -> torch.Tensor:
    if args.input_images:
        batches = [
            preprocess_image(
                image_path=path,
                image_size=args.image_size,
                mean=preprocess["mean"],
                std=preprocess["std"],
            )
            for path in args.input_images
        ]
        return torch.cat(batches, dim=0)

    return torch.randn(args.synthetic_batch, 3, args.image_size, args.image_size)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, preprocess = load_exported_model(args.export_path, device)

    images = _load_batch(args, preprocess)
    triplane = chunked_inference(
        model=model,
        batched_images=images,
        chunk_size=args.chunk_size,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(triplane, args.output)

    print(f"device={device.type}")
    print(f"batch={images.shape[0]} chunk_size={args.chunk_size}")
    print(f"output_shape={tuple(triplane.shape)}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
