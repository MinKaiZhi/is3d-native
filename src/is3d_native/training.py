from __future__ import annotations

import glob
import math
import os
from dataclasses import asdict, dataclass
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
    weight_decay: float = 0.01
    grad_accum_steps: int = 4
    grad_clip_norm: float = 1.0
    lr_warmup_steps: int = 50
    lr_min_ratio: float = 0.2
    use_cosine_lr: bool = True

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
    token_mask_start_ratio: float = 0.0
    token_mask_warmup_steps: int = 0

    train_shards: str | None = None
    train_key_include: str | None = "__images__"
    dataloader_workers: int = 4
    wds_shuffle_buffer: int = 1024


@dataclass
class TrainState:
    step: int = 0
    update_step: int = 0
    current_lr: float = 0.0


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


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _scheduled_lr(config: TrainConfig, update_step: int, total_updates: int) -> float:
    if not config.use_cosine_lr:
        return config.lr

    warmup = max(0, min(config.lr_warmup_steps, total_updates))
    min_lr = config.lr * config.lr_min_ratio

    if warmup > 0 and update_step <= warmup:
        return config.lr * (update_step / warmup)

    if total_updates <= warmup:
        return config.lr

    progress = (update_step - warmup) / max(1, total_updates - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (config.lr - min_lr) * cosine


def _scheduled_token_mask_ratio(config: TrainConfig, step: int) -> float:
    target = config.triplane_token_mask_ratio
    start = config.token_mask_start_ratio
    warmup = max(0, min(config.token_mask_warmup_steps, config.steps))

    if warmup <= 0:
        return target
    if step > warmup:
        return target

    progress = (step - 1) / max(1, warmup)
    progress = min(max(progress, 0.0), 1.0)
    return start + (target - start) * progress


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _serialize_train_config(config: TrainConfig) -> dict[str, object]:
    payload = asdict(config)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _optimizer_step(
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model: IS3DModel,
    config: TrainConfig,
    device: torch.device,
) -> None:
    if config.grad_clip_norm > 0:
        if device.type == "cuda":
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

    if device.type == "cuda":
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _build_checkpoint_payload(
    *,
    model: IS3DModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    state: TrainState,
) -> dict[str, object]:
    rng_state: dict[str, object] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()

    return {
        "format_version": 2,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "train_state": {
            "step": state.step,
            "update_step": state.update_step,
            "current_lr": state.current_lr,
        },
        "train_config": _serialize_train_config(config),
        "rng_state": rng_state,
    }


def _restore_rng_state(payload: dict[str, object], device: torch.device) -> None:
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        return

    cpu_state = rng_state.get("cpu")
    if isinstance(cpu_state, torch.Tensor):
        torch.set_rng_state(cpu_state)

    cuda_state = rng_state.get("cuda")
    if device.type == "cuda" and isinstance(cuda_state, list) and cuda_state:
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except RuntimeError as exc:
            print(f"Warning: failed to restore CUDA RNG state: {exc}")


def _load_resume_state(
    *,
    checkpoint_path: Path,
    model: IS3DModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    device: torch.device,
    total_updates: int,
) -> TrainState:
    payload = _safe_torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(
            f"Checkpoint must be a dict or contain 'state_dict'. Got type={type(payload)}"
        )
    if "state_dict" not in payload:
        raise KeyError("Checkpoint must contain key 'state_dict'.")

    model.load_state_dict(payload["state_dict"], strict=True)

    optimizer_loaded = False
    optimizer_state = payload.get("optimizer")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
        optimizer_loaded = True

    scaler_loaded = False
    scaler_state = payload.get("scaler")
    if isinstance(scaler_state, dict) and scaler_state:
        scaler.load_state_dict(scaler_state)
        scaler_loaded = True

    train_state_payload = payload.get("train_state")
    if not isinstance(train_state_payload, dict):
        train_state_payload = {}

    step = int(train_state_payload.get("step", 0))
    if step < 0:
        raise ValueError(f"Invalid resume step: {step}")
    if step > config.steps:
        raise ValueError(
            f"Checkpoint step ({step}) is greater than target steps ({config.steps})."
        )

    default_update_step = math.ceil(step / config.grad_accum_steps) if step > 0 else 0
    update_step = int(train_state_payload.get("update_step", default_update_step))
    update_step = max(0, update_step)

    current_lr = train_state_payload.get("current_lr")
    if current_lr is None:
        if update_step > 0:
            current_lr = _scheduled_lr(config, update_step, total_updates)
        else:
            current_lr = config.lr
    current_lr = float(current_lr)
    if current_lr <= 0:
        current_lr = config.lr

    _set_optimizer_lr(optimizer, current_lr)
    _restore_rng_state(payload, device)

    if not optimizer_loaded:
        print("Resume note: optimizer state missing in checkpoint, using fresh optimizer state.")
    if device.type == "cuda" and not scaler_loaded:
        print("Resume note: GradScaler state missing in checkpoint, using fresh scaler state.")

    return TrainState(step=step, update_step=update_step, current_lr=current_lr)


def run_training_loop(
    config: TrainConfig,
    resume_from_checkpoint: Path | None = None,
    output_checkpoint: Path | None = None,
) -> IS3DModel:
    if config.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    if config.lr <= 0:
        raise ValueError("lr must be > 0")
    if not (0.0 <= config.lr_min_ratio <= 1.0):
        raise ValueError("lr_min_ratio must be in [0, 1]")
    if config.token_mask_warmup_steps < 0:
        raise ValueError("token_mask_warmup_steps must be >= 0")
    if not (0.0 <= config.token_mask_start_ratio <= 1.0):
        raise ValueError("token_mask_start_ratio must be in [0, 1]")
    if not (0.0 <= config.triplane_token_mask_ratio <= 1.0):
        raise ValueError("triplane_token_mask_ratio must be in [0, 1]")

    device = _select_device()
    set_runtime_flags(device)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    total_updates = math.ceil(config.steps / config.grad_accum_steps)
    state = TrainState(step=0, update_step=0, current_lr=config.lr)
    if resume_from_checkpoint is not None:
        state = _load_resume_state(
            checkpoint_path=resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            device=device,
            total_updates=total_updates,
        )
        print(
            f"Resume checkpoint loaded: {resume_from_checkpoint} "
            f"(step={state.step}, updates={state.update_step}, lr={state.current_lr:.2e})"
        )
    else:
        _set_optimizer_lr(optimizer, state.current_lr)

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
    accum_steps = 0

    for step in range(state.step + 1, config.steps + 1):
        images = next(data_iter)
        current_mask_ratio = _scheduled_token_mask_ratio(config, step)

        use_amp = config.use_bf16_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                target = model(images, use_checkpoint=False, token_mask_ratio=0.0)

            pred = model(
                images,
                use_checkpoint=config.checkpoint_segments,
                token_mask_ratio=current_mask_ratio,
            )
            loss = nn.functional.mse_loss(pred, target) / config.grad_accum_steps

        if torch.isnan(loss) or torch.isnan(pred).any():
            _save_nan_snapshot(pred, step, config.snapshot_dir)
            raise RuntimeError(f"NaN detected at step {step}. Snapshot saved.")

        if device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_steps += 1
        if accum_steps >= config.grad_accum_steps:
            state.update_step += 1
            state.current_lr = _scheduled_lr(config, state.update_step, total_updates)
            _set_optimizer_lr(optimizer, state.current_lr)
            _optimizer_step(
                optimizer=optimizer,
                scaler=scaler,
                model=model,
                config=config,
                device=device,
            )
            accum_steps = 0

        if step % config.log_every == 0:
            msg = f"[step {step}] loss={(loss.item() * config.grad_accum_steps):.6f}"
            msg += f" | lr={state.current_lr:.2e} | token_mask_ratio={current_mask_ratio:.3f}"
            if device.type == "cuda":
                max_mem = torch.cuda.max_memory_allocated() / 1024**2
                msg += f" | max_mem_mb={max_mem:.2f} | tf32={torch.backends.cuda.matmul.allow_tf32}"
            else:
                msg += f" | device={device.type}"
            print(msg)

        state.step = step

    if accum_steps != 0:
        state.update_step += 1
        state.current_lr = _scheduled_lr(config, state.update_step, total_updates)
        _set_optimizer_lr(optimizer, state.current_lr)
        _optimizer_step(
            optimizer=optimizer,
            scaler=scaler,
            model=model,
            config=config,
            device=device,
        )

    state.step = config.steps
    if output_checkpoint is not None:
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            state=state,
        )
        torch.save(payload, output_checkpoint)
        print(
            f"Saved checkpoint: {output_checkpoint} "
            f"(step={state.step}, updates={state.update_step})"
        )

    return model

