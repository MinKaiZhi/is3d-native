from __future__ import annotations

import argparse
import platform
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CUDA/PyTorch runtime for is3d-native.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when checks fail.")
    parser.add_argument(
        "--expect-cuda-major",
        type=int,
        default=12,
        help="Expected CUDA major version from torch.version.cuda.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    cuda_runtime = torch.version.cuda
    print(f"torch.version.cuda: {cuda_runtime}")

    if cuda_runtime is None:
        failures.append("PyTorch is not built with CUDA.")
    else:
        major = int(cuda_runtime.split(".")[0])
        if major != args.expect_cuda_major:
            failures.append(
                f"Expected CUDA major {args.expect_cuda_major}, got {major}."
            )

    is_cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available: {is_cuda_available}")

    if is_cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for idx in range(device_count):
            name = torch.cuda.get_device_name(idx)
            cc = torch.cuda.get_device_capability(idx)
            print(f"  - cuda:{idx} name={name} capability={cc[0]}.{cc[1]}")
        print(f"TF32 enabled (matmul): {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
    else:
        failures.append("No CUDA device available. Verify NVIDIA driver and CUDA runtime.")

    if failures:
        print("\n[FAILURES]")
        for item in failures:
            print(f"- {item}")
        return 1 if args.strict else 0

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
