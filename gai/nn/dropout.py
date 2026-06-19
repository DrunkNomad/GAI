from .module import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        import torch
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)
