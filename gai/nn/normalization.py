import numpy as np
from ..tensor import Tensor
from .module import Module


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
            self.bias = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x):
        dim = tuple(range(-len(self.normalized_shape), 0)) if self.normalized_shape else None
        mean = x.mean(axis=dim, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=dim, keepdims=True)
        x_norm = (x - mean) / (var + self.eps) ** 0.5
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm
