import torch
import torch.nn.functional as F
from .module import Module


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)


class GELU(Module):
    def forward(self, x):
        return F.gelu(x)


class Sigmoid(Module):
    def forward(self, x):
        return torch.sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        return torch.tanh(x)


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)
