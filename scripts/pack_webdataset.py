from __future__ import annotations

import argparse
import os
from pathlib import Path

import webdataset as wds

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUX_EXTS = {".json", ".txt", ".npy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack image dataset into WebDataset tar shards.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Source dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Target shard directory.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="shard-%06d.tar",
        help="Output tar naming pattern.",
    )
    parser.add_argument("--maxcount", type=int, default=5000, help="Samples per shard.")
    return parser.parse_args()


def iter_image_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def sample_key(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root).with_suffix("")
    # WebDataset parser splits keys at the first '.', so sanitize dots to keep a stable image extension.
    key = str(rel).replace("\\", "/").replace("/", "__").replace(".", "_")
    return key


def build_sample(root: Path, image_path: Path) -> dict[str, bytes | str]:
    image_ext = image_path.suffix.lower().lstrip(".")
    sample = {
        "__key__": sample_key(root, image_path),
        image_ext: image_path.read_bytes(),
    }

    for aux_ext in AUX_EXTS:
        aux_path = image_path.with_suffix(aux_ext)
        if aux_path.exists():
            sample[aux_ext.lstrip(".")] = aux_path.read_bytes()

    return sample


def wds_path_for_writer(output_dir: Path, pattern: str) -> str:
    """
    Convert output shard pattern to a WebDataset-compatible target string.

    On Windows, absolute paths like D:\\... are parsed as URL scheme "d" by
    webdataset.gopen, so we use file:/// URI form.
    """
    if output_dir.is_absolute() and os.name == "nt":
        posix_dir = output_dir.resolve().as_posix().rstrip("/")
        return f"file:///{posix_dir}/{pattern}"
    return str(output_dir / pattern)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_image_files(args.input_dir))
    if not images:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    shard_pattern = wds_path_for_writer(args.output_dir, args.pattern)
    print(f"Packing {len(images)} images into {shard_pattern} (maxcount={args.maxcount})")

    with wds.ShardWriter(shard_pattern, maxcount=args.maxcount) as sink:
        for idx, image_path in enumerate(images, start=1):
            sink.write(build_sample(args.input_dir, image_path))
            if idx % 1000 == 0:
                print(f"Packed {idx}/{len(images)}")

    print("Done.")


if __name__ == "__main__":
    main()
