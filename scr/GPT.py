from dataclasses import dataclass


import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module, Embedding

from .Transformer import Decoder, LayerNorm


@dataclass
class GPTConfig:
    vocab_size: int  # 词汇表大小
    context_length: int  # 上下文长度
    n_layer: int  # 层数
    n_head: int  # 注意力头的数量
    emb_dim: int  # 嵌入维度
    drop_rate: float  # dropout 率
    qkv_bias: bool = False  # 查询-键-值偏置


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
        idx = in_idx.long()
        tok_embeds = self.token_emb(idx)
        pos_ids = torch.arange(seq_len, device=idx.device)
        pos_embeds = self.pos_emb(pos_ids)
        x = tok_embeds + pos_embeds
        x = self.drop(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
