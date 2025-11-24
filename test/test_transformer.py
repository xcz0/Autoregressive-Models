import torch
from scr.Transformer import LayerNorm


def test_layer_norm_normalizes_each_row():
    torch.manual_seed(0)
    batch = torch.randn(2, 5)
    layer_norm = LayerNorm(normalized_shape=5)
    layer_norm.eval()
    normalized = layer_norm(batch)
    mean = normalized.mean(dim=-1, keepdim=True)
    var = normalized.var(dim=-1, unbiased=False, keepdim=True)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)
