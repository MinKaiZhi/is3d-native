from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    import xformers.ops as xops

    HAS_XFORMERS = True
except Exception:
    xops = None
    HAS_XFORMERS = False


def set_runtime_flags(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def attention_with_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
    force_fallback: bool = False,
) -> torch.Tensor:
    """
    Unified attention call with CUDA-optimized path and cross-platform fallback.

    Inputs are expected in [B, H, N, D] layout.
    """
    if not force_fallback and q.device.type == "cuda" and HAS_XFORMERS:
        try:
            q_x = q.permute(0, 2, 1, 3).contiguous()
            k_x = k.permute(0, 2, 1, 3).contiguous()
            v_x = v.permute(0, 2, 1, 3).contiguous()
            out = xops.memory_efficient_attention(
                q_x,
                k_x,
                v_x,
                p=dropout_p if training else 0.0,
            )
            return out.permute(0, 2, 1, 3).contiguous()
        except Exception:
            pass

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )
