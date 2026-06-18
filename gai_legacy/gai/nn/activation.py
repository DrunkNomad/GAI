import numpy as np
from ..tensor import Tensor
from .module import Module


class ReLU(Module):
    def forward(self, x):
        mask = Tensor(np.where(x.data > 0, 1.0, 0.0), requires_grad=False)
        return x * mask


class GELU(Module):
    def forward(self, x):
        return x * Tensor(0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x.data + 0.044715 * x.data ** 3))), requires_grad=False)


class Sigmoid(Module):
    def forward(self, x):
        return Tensor(1.0 / (1.0 + np.exp(-x.data)), requires_grad=False)


class Tanh(Module):
    def forward(self, x):
        return Tensor(np.tanh(x.data), requires_grad=False)


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x_exp = np.exp(x.data - x.data.max(axis=self.dim, keepdims=True))
        out = x_exp / x_exp.sum(axis=self.dim, keepdims=True)
        out_t = Tensor(out, requires_grad=x.requires_grad)
        if out_t.requires_grad:
            class SoftmaxFn:
                @staticmethod
                def backward(depends_on, grad):
                    s = out_t.data
                    return [s * (grad - (s * grad).sum(axis=-1, keepdims=True))]
            out_t._depends_on = [(x, {"dim": self.dim})]
            out_t._creation_op = SoftmaxFn
        return out_t
