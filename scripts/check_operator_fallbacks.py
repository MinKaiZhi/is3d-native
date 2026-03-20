from __future__ import annotations

import argparse

import torch

from is3d_native.models.ops import HAS_XFORMERS, attention_with_fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CUDA-op fallback paths.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=32)
    return parser.parse_args()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = _device()
    print(f"device={device.type} xformers_available={HAS_XFORMERS}")

    bsz = 2
    q = torch.randn(bsz, args.heads, args.seq_len, args.head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out_auto = attention_with_fallback(q, k, v, training=False)
    out_fallback = attention_with_fallback(q, k, v, training=False, force_fallback=True)

    max_abs_err = (out_auto - out_fallback).abs().max().item()
    print(f"auto_shape={tuple(out_auto.shape)} fallback_shape={tuple(out_fallback.shape)}")
    print(f"max_abs_err_vs_fallback={max_abs_err:.6e}")


if __name__ == "__main__":
    main()
