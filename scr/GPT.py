from dataclasses import dataclass


import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module, Embedding

from .Transformer import Decoder, LayerNorm, DecoderConfig


@dataclass
class GPTConfig(DecoderConfig):
    n_layer: int = 12  # 层数


class GPTModel(Module):
    def __init__(self, cfg: GPTConfig):
        """ """
        super().__init__()
        self.token_emb = Embedding(
            num_embeddings=cfg.vocab_size, embedding_dim=cfg.emb_dim
        )
        # 使用可学习的位置编码
        self.pos_emb = Embedding(
            num_embeddings=cfg.context_length, embedding_dim=cfg.emb_dim
        )
        self.drop = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(*[Decoder(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: Tensor):
        batch_size, seq_len = in_idx.shape
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds "
                f"maximum context length ({self.pos_emb.num_embeddings})"
            )
        tok_embeds = self.token_emb(in_idx.long())
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
