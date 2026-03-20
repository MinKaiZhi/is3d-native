from __future__ import annotations

from pathlib import Path

import yaml

from is3d_native.training import TrainConfig


def load_train_config(path: Path) -> TrainConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig(
        batch_size=payload.get("batch_size", 2),
        steps=payload.get("steps", 200),
        lr=payload.get("lr", 1e-4),
        grad_accum_steps=payload.get("grad_accum_steps", 4),
        use_bf16_amp=payload.get("use_bf16_amp", True),
        checkpoint_segments=payload.get("checkpoint_segments", True),
        log_every=payload.get("log_every", 20),
        snapshot_dir=Path(payload.get("snapshot_dir", "artifacts/nan_snapshots")),
        image_size=payload.get("image_size", 128),
        in_channels=payload.get("in_channels", 3),
        fastvit_embed_dim=payload.get("fastvit_embed_dim", 256),
        fastvit_depth=payload.get("fastvit_depth", 6),
        fastvit_num_heads=payload.get("fastvit_num_heads", 8),
        fastvit_patch_size=payload.get("fastvit_patch_size", 8),
        triplane_depth=payload.get("triplane_depth", 4),
        triplane_num_heads=payload.get("triplane_num_heads", 8),
        triplane_channels=payload.get("triplane_channels", 24),
        triplane_resolution=payload.get("triplane_resolution", 32),
        triplane_token_mask_ratio=payload.get("triplane_token_mask_ratio", 0.15),
        train_shards=payload.get("train_shards"),
        train_key_include=payload.get("train_key_include", "__images__"),
        dataloader_workers=payload.get("dataloader_workers", 4),
        wds_shuffle_buffer=payload.get("wds_shuffle_buffer", 1024),
    )
