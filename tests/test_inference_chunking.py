from __future__ import annotations

import torch
from torch import nn

from is3d_native.inference import chunked_inference


class DummyModel(nn.Module):
    def forward(self, x: torch.Tensor, use_checkpoint: bool = False, token_mask_ratio: float = 0.0) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3), keepdim=True)
        return pooled.unsqueeze(2).expand(x.shape[0], 3, 4, 8, 8)


def test_chunked_inference_matches_full_batch() -> None:
    model = DummyModel().eval()
    batch = torch.randn(5, 3, 16, 16)

    out_full = chunked_inference(model, batch, chunk_size=5, device=torch.device("cpu"))
    out_chunk = chunked_inference(model, batch, chunk_size=2, device=torch.device("cpu"))

    assert out_full.shape == out_chunk.shape
    assert torch.allclose(out_full, out_chunk)
