from dataclasses import dataclass


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module, Embedding


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.shift = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift


class Decoder(Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class Encoder(Module):
    pass
