from __future__ import annotations

import io

import torch

from is3d_native.inference import DEFAULT_MEAN, DEFAULT_STD, load_exported_model
from is3d_native.training import TrainConfig, build_model


def _model_config_from_train_cfg(cfg: TrainConfig) -> dict:
    return {
        "fastvit": {
            "in_channels": cfg.in_channels,
            "embed_dim": cfg.fastvit_embed_dim,
            "depth": cfg.fastvit_depth,
            "num_heads": cfg.fastvit_num_heads,
            "mlp_ratio": 4.0,
            "patch_size": cfg.fastvit_patch_size,
            "attn_dropout": 0.0,
        },
        "triplane": {
            "embed_dim": cfg.fastvit_embed_dim,
            "depth": cfg.triplane_depth,
            "num_heads": cfg.triplane_num_heads,
            "mlp_ratio": 4.0,
            "plane_channels": cfg.triplane_channels,
            "plane_resolution": cfg.triplane_resolution,
            "attn_dropout": 0.0,
            "token_mask_ratio": cfg.triplane_token_mask_ratio,
        },
    }


def _small_train_cfg() -> TrainConfig:
    return TrainConfig(
        fastvit_embed_dim=32,
        fastvit_depth=1,
        fastvit_num_heads=4,
        fastvit_patch_size=8,
        triplane_depth=1,
        triplane_num_heads=4,
        triplane_channels=4,
        triplane_resolution=8,
    )


def _payload_to_buffer(payload: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    return buffer


def test_export_bundle_can_be_loaded() -> None:
    cfg = _small_train_cfg()
    model = build_model(cfg).eval()

    payload = {
        "state_dict": model.state_dict(),
        "model_config": _model_config_from_train_cfg(cfg),
        "preprocess": {"mean": DEFAULT_MEAN, "std": DEFAULT_STD},
    }
    export_buffer = _payload_to_buffer(payload)

    loaded_model, preprocess = load_exported_model(export_buffer, torch.device("cpu"))
    assert preprocess["mean"] == DEFAULT_MEAN
    assert preprocess["std"] == DEFAULT_STD

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = loaded_model(x, use_checkpoint=False, token_mask_ratio=0.0)
    assert y.ndim == 5


def test_export_bundle_deploy_state_dict_autodetect() -> None:
    cfg = _small_train_cfg()
    model = build_model(cfg).eval()
    model.switch_to_deploy()

    payload = {
        # Simulate old export files that have deploy weights but no fastvit.deploy flag.
        "state_dict": model.state_dict(),
        "model_config": _model_config_from_train_cfg(cfg),
        "preprocess": {"mean": DEFAULT_MEAN, "std": DEFAULT_STD},
    }
    export_buffer = _payload_to_buffer(payload)

    loaded_model, _ = load_exported_model(export_buffer, torch.device("cpu"))

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = loaded_model(x, use_checkpoint=False, token_mask_ratio=0.0)
    assert y.ndim == 5
