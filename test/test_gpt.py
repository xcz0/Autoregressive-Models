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


def test_generate_increases_sequence_length():
    torch.manual_seed(2)
    model = _tiny_gpt_model()
    vocab_size = model.token_emb.num_embeddings
    context_length = model.pos_emb.num_embeddings
    initial_idx = torch.randint(0, vocab_size, (1, 5))
    max_new_tokens = 10

    generated = model.generate(initial_idx, max_new_tokens, context_length)

    assert generated.shape == (1, 5 + max_new_tokens)


def test_generate_respects_context_size():
    torch.manual_seed(3)
    model = _tiny_gpt_model()
    vocab_size = model.token_emb.num_embeddings
    # 创建一个超过context_size的初始序列
    initial_idx = torch.randint(0, vocab_size, (1, 20))
    context_size = 10
    max_new_tokens = 5

    generated = model.generate(initial_idx, max_new_tokens, context_size)

    assert generated.shape == (1, 20 + max_new_tokens)


def test_generate_output_is_valid_token_indices():
    torch.manual_seed(4)
    model = _tiny_gpt_model()
    vocab_size = model.token_emb.num_embeddings
    context_length = model.pos_emb.num_embeddings
    initial_idx = torch.randint(0, vocab_size, (2, 8))
    max_new_tokens = 15

    generated = model.generate(initial_idx, max_new_tokens, context_length)

    assert torch.all(generated >= 0)
    assert torch.all(generated < vocab_size)


def test_generate_preserves_initial_sequence():
    torch.manual_seed(5)
    model = _tiny_gpt_model()
    vocab_size = model.token_emb.num_embeddings
    context_length = model.pos_emb.num_embeddings
    initial_idx = torch.randint(0, vocab_size, (1, 7))
    max_new_tokens = 3

    generated = model.generate(initial_idx, max_new_tokens, context_length)

    assert torch.equal(generated[:, :7], initial_idx)


def test_generate_is_stochastic_with_different_seeds():
    vocab_size = 50
    initial_idx = torch.randint(0, vocab_size, (1, 5))
    max_new_tokens = 10

    torch.manual_seed(6)
    model1 = _tiny_gpt_model()
    generated1 = model1.generate(
        initial_idx, max_new_tokens, model1.pos_emb.num_embeddings
    )

    torch.manual_seed(7)
    model2 = _tiny_gpt_model()
    generated2 = model2.generate(
        initial_idx, max_new_tokens, model2.pos_emb.num_embeddings
    )

    assert not torch.equal(generated1, generated2)
