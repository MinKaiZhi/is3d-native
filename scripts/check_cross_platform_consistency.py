from __future__ import annotations

import argparse
from pathlib import Path

import torch

from is3d_native.inference import (
    choose_device,
    chunked_inference,
    load_exported_model,
    safe_torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-platform triplane consistency check.")
    parser.add_argument("--export-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=1e-5)

    parser.add_argument("--input-tensor", type=Path, default=None)
    parser.add_argument("--save-input", type=Path, default=None)

    parser.add_argument("--reference-tensor", type=Path, default=None)
    parser.add_argument("--save-output", type=Path, default=Path("artifacts/consistency/current_output.pt"))
    return parser.parse_args()


def _load_or_create_input(args: argparse.Namespace) -> torch.Tensor:
    if args.input_tensor is not None:
        tensor = safe_torch_load(args.input_tensor, map_location="cpu")
        return tensor.float()

    torch.manual_seed(args.seed)
    tensor = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    return tensor


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    model, _ = load_exported_model(args.export_path, device)
    inputs = _load_or_create_input(args)

    if args.save_input is not None:
        args.save_input.parent.mkdir(parents=True, exist_ok=True)
        torch.save(inputs.cpu(), args.save_input)
        print(f"saved_input={args.save_input}")

    outputs = chunked_inference(
        model=model,
        batched_images=inputs,
        chunk_size=args.chunk_size,
        device=device,
    )

    args.save_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, args.save_output)
    print(f"saved_output={args.save_output}")
    print(f"device={device.type} shape={tuple(outputs.shape)} chunk_size={args.chunk_size}")

    if args.reference_tensor is None:
        print("reference_tensor not provided; generated output only.")
        return

    reference = safe_torch_load(args.reference_tensor, map_location="cpu").float()
    if reference.shape != outputs.shape:
        raise ValueError(f"Shape mismatch: reference={tuple(reference.shape)} current={tuple(outputs.shape)}")

    diff = (outputs.float() - reference).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    print(f"max_abs_err={max_abs:.6e}")
    print(f"mean_abs_err={mean_abs:.6e}")
    if max_abs >= args.threshold:
        raise SystemExit(
            f"FAILED: max_abs_err={max_abs:.6e} >= threshold={args.threshold:.6e}"
        )
    print(f"PASSED: max_abs_err < {args.threshold:.6e}")


if __name__ == "__main__":
    main()
