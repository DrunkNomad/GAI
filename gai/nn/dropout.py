import numpy as np
from ..tensor import Tensor
from .module import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = np.random.binomial(1, 1.0 - self.p, x.shape).astype(np.float32) / (1.0 - self.p)
        return x * Tensor(mask, requires_grad=False)
