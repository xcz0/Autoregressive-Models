import torch
from scr.Transformer import LayerNorm, FeedForward, Decoder, DecoderConfig


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


def test_feedforward_output_shape_matches_input():
    torch.manual_seed(0)
    emb_dim = 256
    batch_size = 4
    seq_len = 10

    feedforward = FeedForward(emb_dim=emb_dim)
    feedforward.eval()

    input_tensor = torch.randn(batch_size, seq_len, emb_dim)
    output_tensor = feedforward(input_tensor)

    assert output_tensor.shape == input_tensor.shape


def test_decoder_output_shape_matches_input():
    torch.manual_seed(0)
    batch_size = 4
    seq_len = 10
    emb_dim = 256
    n_head = 8

    cfg = DecoderConfig(
        vocab_size=50000,
        context_length=seq_len,
        n_head=n_head,
        emb_dim=emb_dim,
        drop_rate=0.1,
        qkv_bias=False,
    )
    decoder = Decoder(cfg)
    decoder.eval()

    input_tensor = torch.randn(batch_size, seq_len, emb_dim)
    output_tensor = decoder(input_tensor)

    assert output_tensor.shape == input_tensor.shape
