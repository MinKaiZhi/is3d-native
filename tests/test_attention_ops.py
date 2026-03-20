import torch

from is3d_native.models.ops import attention_with_fallback


def test_attention_fallback_cpu_shape() -> None:
    q = torch.randn(2, 4, 16, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = attention_with_fallback(q, k, v, force_fallback=True, training=False)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
