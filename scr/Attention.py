import torch
from torch import Tensor
from torch.nn import Dropout, Linear, Module

from einops import einsum, rearrange


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
        self.W_query = Linear(in_features=in_dim, out_features=out_dim, bias=qkv_bias)
        self.W_key = Linear(in_features=in_dim, out_features=out_dim, bias=qkv_bias)
        self.W_value = Linear(in_features=in_dim, out_features=out_dim, bias=qkv_bias)
        self.dropout = Dropout(p=dropout)
        if mask:
            self.register_buffer(
                name="mask",
                tensor=torch.triu(
                    input=torch.ones(context_length, context_length, dtype=torch.bool),
                    diagonal=1,
                ),
            )

    def _build_mask(self, seq_len: int) -> Tensor | None:
        mask = getattr(self, "mask", None)
        return mask[:seq_len, :seq_len] if mask is not None else None

    def _scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> Tensor:
        """
        手动实现缩放点积注意力机制。

        Args:
            query: 查询张量，形状为 (..., seq_len, head_dim)
            key: 键张量，形状为 (..., seq_len, head_dim)
            value: 值张量，形状为 (..., seq_len, head_dim)
            attn_mask: 注意力掩码，True 表示需要被屏蔽的位置
            dropout_p: Dropout 概率

        Returns:
            注意力输出，形状与 value 相同
        """
        head_dim = query.shape[-1]
        scale = head_dim**-0.5

        # 计算注意力分数: query @ key^T
        attn_scores = (
            einsum(
                query, key, "... seq_q head_dim, ... seq_k head_dim -> ... seq_q seq_k"
            )
            * scale
        )

        # 应用掩码
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        # Softmax 归一化
        attn_weights = torch.softmax(input=attn_scores, dim=-1)

        # 应用 dropout
        if self.training and dropout_p > 0.0:
            attn_weights = torch.dropout(input=attn_weights, p=dropout_p, train=True)

        # 计算输出: attn_weights @ value
        return einsum(
            attn_weights,
            value,
            "... seq_q seq_k, ... seq_k head_dim -> ... seq_q head_dim",
        )

    def forward(self, query_raw: Tensor, key_raw: Tensor, value_raw: Tensor) -> Tensor:
        seq_len = query_raw.shape[-2]
        dropout_p = self.dropout.p if self.training else 0.0
        return self._scaled_dot_product_attention(
            query=self.W_query(query_raw),
            key=self.W_key(key_raw),
            value=self.W_value(value_raw),
            attn_mask=self._build_mask(seq_len=seq_len),
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
        return super().forward(query_raw=x, key_raw=x, value_raw=x)


class CausalAttention(SelfAttention):
    """因果自注意力：带掩码的自注意力，用于自回归模型。"""

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
            qkv_bias=qkv_bias,
        )
        self.dropout = Dropout(p=dropout)


class MultiHeadAttention(Attention):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        context_length: int,
        dropout: float = 0.0,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        mask: bool = False,
        qkv_bias: bool = False,
    ):
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_kv_heads <= num_heads and num_heads % num_kv_heads == 0
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            context_length=context_length,
            mask=mask,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = out_dim // num_heads
        self.out_proj = Linear(in_features=self.model_dim, out_features=self.model_dim)

    def _split_heads(self, x: Tensor, heads: int) -> Tensor:
        """将张量分割为多头形式: (batch, seq_len, out_dim) -> (batch, num_heads, seq_len, head_dim)"""
        return rearrange(
            tensor=x,
            pattern="batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim",
            num_heads=heads,
        )

    def forward(self, x: Tensor) -> Tensor:
        query = self._split_heads(x=self.W_query(x), heads=self.num_heads)
        key = self._split_heads(x=self.W_key(x), heads=self.num_kv_heads)
        value = self._split_heads(x=self.W_value(x), heads=self.num_kv_heads)
        seq_len = x.shape[-2]
        attn_mask = self._build_mask(seq_len=seq_len)
        dropout_p = self.dropout.p if self.training else 0.0
        context_vec = self._scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
        context_vec = rearrange(
            tensor=context_vec,
            pattern="batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)",
        )
        return self.out_proj(context_vec)
