import torch
from torch import Tensor
from torch.nn import Module, Parameter


class LayerNorm(Module):
    """
    实现层归一化（Layer Normalization）。将每一个样本的特征向量 $x$ 转变为均值为 0，标准差为 1 的特征向量
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = Parameter(torch.ones(normalized_shape))
        self.shift = Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift


class RMSNorm(Module):
    """
    实现 RMS 归一化（Root Mean Square Normalization）。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed
