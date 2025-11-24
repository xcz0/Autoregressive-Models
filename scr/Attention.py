import torch
from torch import Tensor
from torch.nn import Dropout, Linear, Module
from torch.nn import functional as F

from einops import rearrange


class Attention(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        context_length: int,
        mask: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.model_dim = out_dim
        self.W_query = Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_key = Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_value = Linear(in_dim, out_dim, bias=qkv_bias)
        self.dropout = Dropout(dropout)
        if mask:
            self.register_buffer(
                "mask",
                torch.triu(
                    torch.ones(context_length, context_length, dtype=torch.bool),
                    diagonal=1,
                ),
            )

    def _build_mask(self, seq_len: int) -> Tensor | None:
        mask = getattr(self, "mask", None)
        if mask is None:
            return None
        return mask[:seq_len, :seq_len]

    def forward(self, query_raw: Tensor, key_raw: Tensor, value_raw: Tensor) -> Tensor:
        query = self.W_query(query_raw)
        key = self.W_key(key_raw)
        value = self.W_value(value_raw)
        seq_len = query.shape[-2]
        attn_mask = self._build_mask(seq_len)
        dropout_p = self.dropout.p if self.training else 0.0
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )


class SelfAttention(Attention):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        context_length: int,
        mask: bool = False,
        qkv_bias: bool = False,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            context_length=context_length,
            mask=mask,
            dropout=0.0,
            qkv_bias=qkv_bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x, x, x)


class CausalAttention(Attention):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            context_length=context_length,
            mask=True,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x, x, x)


class MultiHeadAttention(Attention):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        context_length: int,
        dropout: float = 0.0,
        num_heads: int = 4,
        mask: bool = False,
        qkv_bias: bool = False,
    ):
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            context_length=context_length,
            mask=mask,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.out_proj = Linear(self.model_dim, self.model_dim)

    def forward(self, x: Tensor) -> Tensor:
        query = rearrange(
            self.W_query(x),
            "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        key = rearrange(
            self.W_key(x),
            "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        value = rearrange(
            self.W_value(x),
            "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim",
            num_heads=self.num_heads,
        )
        seq_len = query.shape[-2]
        attn_mask = self._build_mask(seq_len)
        dropout_p = self.dropout.p if self.training else 0.0
        context_vec = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        context_vec = rearrange(
            context_vec,
            "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)",
        )
        return self.out_proj(context_vec)
