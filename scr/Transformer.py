from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import Module

from .Attention import MultiHeadAttention
from .Norm import LayerNorm


@dataclass
class DecoderConfig:
    vocab_size: int  # 词汇表大小
    context_length: int  # 上下文长度
    n_head: int  # 注意力头的数量
    emb_dim: int  # 嵌入维度
    drop_rate: float  # dropout 率
    qkv_bias: bool = False  # 查询-键-值偏置


class FeedForward(Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Decoder(Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            in_dim=cfg.emb_dim,
            out_dim=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_head,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
            mask=True,
        )
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.ffn = FeedForward(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        """
        实现 Transformer 解码器层的前向传播。
        参数：
            x (Tensor): 输入张量，形状为 (batch_size, seq_len, emb_dim)。
        返回：
            Tensor: 输出张量，形状与输入相同。
        说明：层归一化（LayerNorm）应用于这两个组件之前，而 dropout 应用于这两个组件之后，以便对模型进行正则化并防止过拟合。这种方法也被称为前层归一化（Pre-LayerNorm）。
        较早的架构（如最初的 Transformer 模型）在自注意力和前馈神经网络之后才应用层归一化，这种方法被称为后层归一化（Post-LayerNorm），这通常会导致较差的训练效果。
        """
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class Encoder(Module):
    pass
