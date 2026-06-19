import math
import torch
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        k = math.sqrt(1.0 / in_features)
        torch.nn.init.uniform_(self.linear.weight, -k, k)
        if bias:
            torch.nn.init.uniform_(self.linear.bias, -k, k)

    def forward(self, x):
        return self.linear(x)
