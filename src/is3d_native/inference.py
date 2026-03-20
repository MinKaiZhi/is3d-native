from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_image

from is3d_native.models import FastViTConfig, IS3DModel, IS3DModelConfig, TriPlaneConfig

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def choose_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _build_model_from_export_config(model_cfg: dict) -> tuple[IS3DModel, bool]:
    fastvit_payload = dict(model_cfg["fastvit"])
    fastvit_deploy = bool(fastvit_payload.pop("deploy", False))
    fastvit_cfg = FastViTConfig(**fastvit_payload)
    triplane_cfg = TriPlaneConfig(**model_cfg["triplane"])
    model = IS3DModel(IS3DModelConfig(fastvit=fastvit_cfg, triplane=triplane_cfg))
    return model, fastvit_deploy


def _state_dict_uses_fastvit_reparam(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(".mlp.fc1.reparam." in key or ".mlp.fc2.reparam." in key for key in state_dict)


def load_exported_model(export_path: Path, device: torch.device) -> tuple[IS3DModel, dict]:
    payload = safe_torch_load(export_path, map_location="cpu")
    if "state_dict" not in payload or "model_config" not in payload:
        raise KeyError("Export file must contain 'state_dict' and 'model_config'.")

    state_dict = payload["state_dict"]
    model, fastvit_deploy = _build_model_from_export_config(payload["model_config"])
    if fastvit_deploy or _state_dict_uses_fastvit_reparam(state_dict):
        model.switch_to_deploy()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    preprocess = payload.get("preprocess", {"mean": DEFAULT_MEAN, "std": DEFAULT_STD})
    return model, preprocess


def preprocess_image(
    image_path: Path,
    image_size: int,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    img = read_image(str(image_path)).float() / 255.0
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.shape[0] > 3:
        img = img[:3]

    img = img.unsqueeze(0)
    img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)

    mean_t = torch.tensor(mean, dtype=img.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype).view(1, 3, 1, 1)
    return (img - mean_t) / std_t


def chunked_inference(
    model: IS3DModel,
    batched_images: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, batched_images.shape[0], chunk_size):
            chunk = batched_images[start : start + chunk_size].to(device)
            out = model(chunk, use_checkpoint=False, token_mask_ratio=0.0)
            outputs.append(out.cpu())

    return torch.cat(outputs, dim=0)
