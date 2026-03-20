from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import webdataset as wds
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.io import decode_image

from is3d_native.models import FastViTConfig, IS3DModel, IS3DModelConfig, TriPlaneConfig
from is3d_native.models.ops import set_runtime_flags


@dataclass
class TrainConfig:
    batch_size: int = 2
    steps: int = 200
    lr: float = 1e-4
    grad_accum_steps: int = 4
    use_bf16_amp: bool = True
    checkpoint_segments: bool = True
    log_every: int = 20
    snapshot_dir: Path = Path("artifacts/nan_snapshots")

    image_size: int = 128
    in_channels: int = 3

    fastvit_embed_dim: int = 256
    fastvit_depth: int = 6
    fastvit_num_heads: int = 8
    fastvit_patch_size: int = 8

    triplane_depth: int = 4
    triplane_num_heads: int = 8
    triplane_channels: int = 24
    triplane_resolution: int = 32
    triplane_token_mask_ratio: float = 0.15

    train_shards: str | None = None
    train_key_include: str | None = "__images__"
    dataloader_workers: int = 4
    wds_shuffle_buffer: int = 1024


def _save_nan_snapshot(tensor: torch.Tensor, step: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"nan_step_{step:06d}.pt"
    torch.save({"step": step, "tensor": tensor.detach().cpu()}, path)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _sample_key_contains(sample: dict[str, object], key_filter: str) -> bool:
    return key_filter in str(sample.get("__key__", ""))


def _collate_keep_list(batch: list[torch.Tensor]) -> list[torch.Tensor]:
    # Keep variable-size images as a list; we resize+stack in the main process.
    return batch


def _resolve_wds_shards(source_pattern: str) -> list[str]:
    matches = sorted(glob.glob(source_pattern))
    if not matches:
        raise FileNotFoundError(f"No tar shards matched pattern: {source_pattern}")

    if os.name != "nt":
        return matches

    cwd = os.getcwd()
    urls: list[str] = []
    for path in matches:
        try:
            urls.append(os.path.relpath(path, cwd))
        except ValueError:
            urls.append(path)
    return urls


def _prepare_single_image(image: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(image.shape)}")

    if image.shape[0] != config.in_channels:
        if image.shape[0] == 1 and config.in_channels == 3:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 3 and config.in_channels == 1:
            image = image.mean(dim=0, keepdim=True)
        else:
            raise ValueError(
                f"Input channel mismatch: got {image.shape[0]}, expected {config.in_channels}"
            )

    if image.shape[-2:] != (config.image_size, config.image_size):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(config.image_size, config.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image


def _prepare_images(
    images: torch.Tensor | list[torch.Tensor],
    config: TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(images, list):
        prepared = [_prepare_single_image(img, config) for img in images]
        batch = torch.stack(prepared, dim=0)
    else:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape={tuple(images.shape)}")
        prepared = [_prepare_single_image(img, config) for img in images]
        batch = torch.stack(prepared, dim=0)

    return batch.to(device, non_blocking=(device.type == "cuda"))


def _build_train_loader(config: TrainConfig) -> DataLoader | None:
    if not config.train_shards:
        return None

    shard_urls = _resolve_wds_shards(config.train_shards)
    dataset = wds.WebDataset(shard_urls, shardshuffle=False, cache_size=0)

    if config.wds_shuffle_buffer > 0:
        dataset = dataset.shuffle(config.wds_shuffle_buffer)

    if config.train_key_include:
        dataset = dataset.select(partial(_sample_key_contains, key_filter=config.train_key_include))

    dataset = dataset.map(_extract_image_tensor)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.dataloader_workers > 0,
        collate_fn=_collate_keep_list,
    )


def _iter_train_images(config: TrainConfig, device: torch.device) -> Iterator[torch.Tensor]:
    loader = _build_train_loader(config)
    if loader is None:
        while True:
            yield torch.randn(
                config.batch_size,
                config.in_channels,
                config.image_size,
                config.image_size,
                device=device,
            )

    assert loader is not None
    while True:
        for batch in loader:
            yield _prepare_images(batch, config, device)


def build_model(config: TrainConfig) -> IS3DModel:
    model_cfg = IS3DModelConfig(
        fastvit=FastViTConfig(
            in_channels=config.in_channels,
            embed_dim=config.fastvit_embed_dim,
            depth=config.fastvit_depth,
            num_heads=config.fastvit_num_heads,
            patch_size=config.fastvit_patch_size,
        ),
        triplane=TriPlaneConfig(
            embed_dim=config.fastvit_embed_dim,
            depth=config.triplane_depth,
            num_heads=config.triplane_num_heads,
            plane_channels=config.triplane_channels,
            plane_resolution=config.triplane_resolution,
            token_mask_ratio=config.triplane_token_mask_ratio,
        ),
    )
    return IS3DModel(model_cfg)


def run_training_loop(config: TrainConfig) -> IS3DModel:
    device = _select_device()
    set_runtime_flags(device)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if config.train_shards:
        print(
            "Training source: "
            f"{config.train_shards} | key_filter={config.train_key_include} | "
            f"workers={config.dataloader_workers}"
        )
    else:
        print("Training source: synthetic random tensors")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_iter = _iter_train_images(config, device)

    for step in range(1, config.steps + 1):
        images = next(data_iter)

        use_amp = config.use_bf16_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                target = model(images, use_checkpoint=False, token_mask_ratio=0.0)

            pred = model(
                images,
                use_checkpoint=config.checkpoint_segments,
                token_mask_ratio=config.triplane_token_mask_ratio,
            )
            loss = nn.functional.mse_loss(pred, target) / config.grad_accum_steps

        if torch.isnan(loss) or torch.isnan(pred).any():
            _save_nan_snapshot(pred, step, config.snapshot_dir)
            raise RuntimeError(f"NaN detected at step {step}. Snapshot saved.")

        if device.type == "cuda":
            scaler.scale(loss).backward()
            if step % config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if step % config.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if step % config.log_every == 0:
            msg = f"[step {step}] loss={(loss.item() * config.grad_accum_steps):.6f}"
            msg += f" | token_mask_ratio={config.triplane_token_mask_ratio:.2f}"
            if device.type == "cuda":
                max_mem = torch.cuda.max_memory_allocated() / 1024**2
                msg += f" | max_mem_mb={max_mem:.2f} | tf32={torch.backends.cuda.matmul.allow_tf32}"
            else:
                msg += f" | device={device.type}"
            print(msg)

    return model
