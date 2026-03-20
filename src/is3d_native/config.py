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
        weight_decay=payload.get("weight_decay", 0.01),
        grad_accum_steps=payload.get("grad_accum_steps", 4),
        grad_clip_norm=payload.get("grad_clip_norm", 1.0),
        lr_warmup_steps=payload.get("lr_warmup_steps", 50),
        lr_min_ratio=payload.get("lr_min_ratio", 0.2),
        use_cosine_lr=payload.get("use_cosine_lr", True),
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
        token_mask_start_ratio=payload.get("token_mask_start_ratio", 0.0),
        token_mask_warmup_steps=payload.get("token_mask_warmup_steps", 0),
        train_shards=payload.get("train_shards"),
        train_key_include=payload.get("train_key_include", "__images__"),
        dataloader_workers=payload.get("dataloader_workers", 4),
        wds_shuffle_buffer=payload.get("wds_shuffle_buffer", 1024),
    )
