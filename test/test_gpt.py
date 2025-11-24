import torch

from scr.GPT import GPTConfig, GPTModel


def _tiny_gpt_model():
    cfg = GPTConfig(
        vocab_size=50,
        context_length=64,
        emb_dim=16,
        n_head=2,
        n_layer=2,
        drop_rate=0.1,
        qkv_bias=False,
    )
    model = GPTModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    return model


def test_gpt_model_output_shape_matches_vocab():
    torch.manual_seed(0)
    model = _tiny_gpt_model().eval()
    batch = torch.randint(0, model.token_emb.num_embeddings, (3, 12))
    logits = model(batch)

    assert logits.shape == (3, 12, model.token_emb.num_embeddings)


def test_gpt_model_logits_are_deterministic_in_eval_mode():
    torch.manual_seed(1)
    model = _tiny_gpt_model().eval()
    batch = torch.randint(0, model.token_emb.num_embeddings, (2, 10))
    first_pass = model(batch)
    second_pass = model(batch)

    assert torch.allclose(first_pass, second_pass)
