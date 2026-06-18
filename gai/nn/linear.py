import numpy as np
from ..tensor import Tensor
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = np.sqrt(1.0 / in_features)
        self.weight = Tensor(
            np.random.uniform(-k, k, (out_features, in_features)),
            requires_grad=True,
        )
        if bias:
            self.bias = Tensor(
                np.random.uniform(-k, k, (out_features,)),
                requires_grad=True,
            )
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
