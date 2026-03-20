from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from is3d_native.config import load_train_config
from is3d_native.inference import DEFAULT_MEAN, DEFAULT_STD, safe_torch_load
from is3d_native.training import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to state_dict-only bundle.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--input-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint containing state_dict or raw state dict.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/export/is3d_state_dict.pt"),
        help="Output path for exported state_dict bundle.",
    )
    parser.add_argument("--switch-to-deploy", action="store_true")
    return parser.parse_args()


def _state_dict_uses_fastvit_reparam(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(".mlp.fc1.reparam." in key or ".mlp.fc2.reparam." in key for key in state_dict)


def _model_config_dict(cfg, fastvit_deploy: bool) -> dict:
    return {
        "fastvit": {
            "in_channels": cfg.in_channels,
            "embed_dim": cfg.fastvit_embed_dim,
            "depth": cfg.fastvit_depth,
            "num_heads": cfg.fastvit_num_heads,
            "mlp_ratio": 4.0,
            "patch_size": cfg.fastvit_patch_size,
            "attn_dropout": 0.0,
            "deploy": fastvit_deploy,
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


def _load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    payload = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    model = build_model(cfg)

    if args.input_checkpoint is not None:
        _load_checkpoint_weights(model, args.input_checkpoint)

    model.eval()
    if args.switch_to_deploy:
        model.switch_to_deploy()

    state_dict = model.state_dict()
    fastvit_deploy = _state_dict_uses_fastvit_reparam(state_dict)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    export_bundle = {
        "state_dict": state_dict,
        "model_config": _model_config_dict(cfg, fastvit_deploy=fastvit_deploy),
        "preprocess": {
            "mean": DEFAULT_MEAN,
            "std": DEFAULT_STD,
        },
        "meta": {
            "format": "is3d-native-export-v1",
            "state_dict_only": True,
            "fastvit_deploy": fastvit_deploy,
            "train_config": asdict(cfg),
        },
    }

    torch.save(export_bundle, args.output)
    print(f"Exported state_dict bundle: {args.output}")
    print(f"Tensor params: {len(export_bundle['state_dict'])}")


if __name__ == "__main__":
    main()
