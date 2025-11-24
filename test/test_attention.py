import torch
from scr.Attention import MultiHeadAttention


def test_multi_head_attention_forward_is_deterministic():
    torch.manual_seed(123)
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ]
    )
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]
    mha = MultiHeadAttention(3, 4, num_heads=2, context_length=context_length)
    mha.eval()
    output = mha(batch)
    assert output.shape == (2, context_length, 4)
    assert torch.isfinite(output).all()
    assert torch.allclose(output[0], output[1])
