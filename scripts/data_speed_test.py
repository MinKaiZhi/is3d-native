from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image, read_image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.files = [
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not self.files:
            raise FileNotFoundError(f"No image files found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        x = read_image(str(self.files[index])).float() / 255.0
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure DataLoader throughput on Windows.")
    parser.add_argument("--source", type=Path, required=True, help="Image folder or .tar/.tar pattern.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def _to_wds_local_path(path: Path) -> str:
    p = path.resolve()
    if os.name == "nt":
        try:
            return os.path.relpath(str(p), os.getcwd())
        except ValueError:
            return str(p)
    return str(p)


def _resolve_wds_shards(source_raw: str) -> list[str]:
    if "*" in source_raw:
        matches = sorted(glob.glob(source_raw))
        if not matches:
            raise FileNotFoundError(f"No tar shards matched pattern: {source_raw}")
        return [_to_wds_local_path(Path(m)) for m in matches]

    src_path = Path(source_raw)
    if src_path.exists() and src_path.is_file() and src_path.suffix == ".tar":
        return [_to_wds_local_path(src_path)]

    if "{" in source_raw and "}" in source_raw:
        return [source_raw]

    raise FileNotFoundError(f"Expected .tar file or shard pattern, got: {source_raw}")


def _extract_image_tensor(sample: dict[str, object]) -> torch.Tensor:
    for key, value in sample.items():
        if key.startswith("__"):
            continue
        lower = key.lower()
        if lower.endswith(("jpg", "jpeg", "png", "bmp", "webp")):
            if not isinstance(value, (bytes, bytearray)):
                raise TypeError(f"Expected bytes payload for {key}, got {type(value)}")
            encoded = torch.from_numpy(np.frombuffer(value, dtype=np.uint8).copy())
            decoded = decode_image(encoded, mode="RGB")
            return decoded.float() / 255.0
    raise ValueError(f"No image payload found in sample keys: {list(sample.keys())}")


def build_loader(args: argparse.Namespace) -> DataLoader:
    src_raw = str(args.source)

    if src_raw.endswith(".tar") or "{000000.." in src_raw or "*" in src_raw:
        shard_urls = _resolve_wds_shards(src_raw)
        dataset = wds.WebDataset(shard_urls, shardshuffle=False, cache_size=0).map(_extract_image_tensor)
    else:
        dataset = ImageFolderDataset(args.source)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = build_loader(args)

    total_samples = 0
    start = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        if device == "cuda":
            images = images.to("cuda", non_blocking=True)
            _ = images.mean()
            torch.cuda.synchronize()

        total_samples += images.shape[0]

        if batch_idx % args.log_every == 0:
            elapsed = time.perf_counter() - start
            throughput = total_samples / max(elapsed, 1e-6)
            print(
                f"batch={batch_idx} samples={total_samples} elapsed={elapsed:.2f}s "
                f"throughput={throughput:.2f} samples/s"
            )

        if batch_idx >= args.max_batches:
            break

    elapsed = time.perf_counter() - start
    throughput = total_samples / max(elapsed, 1e-6)
    print(f"Done: {total_samples} samples in {elapsed:.2f}s ({throughput:.2f} samples/s)")


if __name__ == "__main__":
    main()
